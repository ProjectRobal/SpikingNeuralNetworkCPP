#include <experimental/simd>
#include <iostream>
#include <string_view>
#include <cmath>

#include "config.hpp"

#include "simd_vector.hpp"

#include "neurons/feedforwardneuron.hpp"

#include "crossovers/onepoint.hpp"
#include "crossovers/fastuniform.hpp"
#include "crossovers/fastonepoint.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/normalized_gauss.hpp"

#include "block.hpp"
 
int main()
{

    snn::SIMDVector a([](size_t x)-> number{ return x;},128);
    snn::SIMDVector b([](size_t x)-> number{ return 2*x;},128);

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);


    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    std::cout<<cross->cross(a,b)<<std::endl;

    gauss->init(a,128);

    a+=0.1;
    
    snn::FeedForwardNeuron<128,4> neuron;

    neuron.setup(norm_gauss);

    neuron.crossover(cross,neuron);

    std::cout<<"Input size: "<<neuron.input_size()<<std::endl;

    snn::Block<snn::FeedForwardNeuron<128,1>,1,32> block(cross);

    block.setup(norm_gauss);

    long double best_reward=-100;

    for(size_t i=0;i<20000;++i)
    {
        block.chooseWorkers();

        snn::SIMDVector output=block.fire(a);

        long double reward=-abs(10.0-output[0]);

        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<std::endl;
            best_reward=reward;
        }

        block.giveReward(-abs(10.0-output[0]));

        if(block.readyToMate())
        {
            std::cout<<"Maiting"<<std::endl;
            block.maiting(norm_gauss);
        }

    }
   
}
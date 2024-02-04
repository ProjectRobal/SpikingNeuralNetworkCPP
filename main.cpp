#include <experimental/simd>
#include <iostream>
#include <string_view>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <numeric>

#include "config.hpp"

#include "simd_vector.hpp"

#include "neurons/feedforwardneuron.hpp"

#include "crossovers/onepoint.hpp"
#include "crossovers/fastuniform.hpp"
#include "crossovers/fastonepoint.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/normalized_gauss.hpp"

#include "mutatiom/gauss_mutation.hpp"

#include "block.hpp"

number stddev(const snn::SIMDVector& vec)
{
    number mean=vec.dot_product();

    snn::SIMDVector omg=vec-mean;

    omg=omg*omg;

    return std::sqrt(omg.dot_product()/vec.size());

}
 
int main()
{

    snn::SIMDVector a([](size_t x)-> number{ return x;},128);
    snn::SIMDVector b([](size_t x)-> number{ return 2*x;},128);

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);

    std::shared_ptr<snn::GaussMutation> mutation=std::make_shared<snn::GaussMutation>(0.f,0.01f,0.1f);


    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    std::cout<<cross->cross(a,b)<<std::endl;

    gauss->init(a,4096);

    a+=0.1;
    
    snn::FeedForwardNeuron<128,4> neuron;

    neuron.setup(norm_gauss);

    neuron.crossover(cross,neuron);

    std::cout<<"Input size: "<<neuron.input_size()<<std::endl;

    snn::Block<snn::FeedForwardNeuron<4096,1>,1,256> block(cross,mutation);

    block.setup(norm_gauss);

    long double best_reward=-100;

    while(abs(best_reward)>0.001f)
    {
        const auto start = std::chrono::steady_clock::now();

        block.chooseWorkers();

        snn::SIMDVector output=block.fire(a);

        //long double reward=-stddev(output);

        long double reward=-abs(20.f-output[0]);

        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<std::endl;
            best_reward=reward;
            std::cout<<"Output: "<<output[0]<<std::endl;
        }

        block.giveReward(reward);

        if(block.readyToMate())
        {
            //std::cout<<"Maiting"<<std::endl;
            block.maiting(norm_gauss);
        }

        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> diff = end - start;

        //std::cout << "Time: " << std::setw(9) << diff.count() << std::endl;

    }
   
}
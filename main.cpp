#include <experimental/simd>
#include <iostream>
#include <string_view>

#include "config.hpp"

#include "simd_vector.hpp"

#include "neurons/feedforwardneuron.hpp"

#include "crossovers/onepoint.hpp"
#include "crossovers/fastuniform.hpp"
#include "crossovers/fastonepoint.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/normalized_gauss.hpp"
 
int main()
{

    snn::SIMDVector a([](size_t x)-> number{ return x;},120);
    snn::SIMDVector b([](size_t x)-> number{ return 2*x;},120);

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,1.f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);


    std::shared_ptr<snn::OnePoint> cross=std::make_shared<snn::OnePoint>();

    //std::cout<<cross->cross(a,b)<<std::endl;

    gauss->init(a,128);

    a+=0.1;
    
    snn::FeedForwardNeuron<128,4> neuron;

    neuron.setup(norm_gauss);

    neuron.crossover(cross,neuron);

    std::cout<<"Input size: "<<neuron.input_size()<<std::endl;

    std::cout<<"Output: "<<neuron.fire(a)<<std::endl;
   
}
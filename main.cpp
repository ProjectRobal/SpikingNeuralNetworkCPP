#include <experimental/simd>
#include <iostream>
#include <string_view>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <chrono>
#include <thread>

#include "config.hpp"

#include "simd_vector.hpp"

#include "initializers/gauss.hpp"
#include "initializers/normalized_gauss.hpp"

#include "encoders/number_encoder.hpp"

#include "FCN.hpp"


number stddev(const snn::SIMDVector& vec)
{
    number mean=vec.dot_product();

    snn::SIMDVector omg=vec-mean;

    omg=omg*omg;

    return std::sqrt(omg.dot_product()/vec.size());

}

using namespace std::chrono_literals;


void callback(size_t i)
{
    std::cout<<"Neuron: "<<i<<" fired"<<std::endl;
}


int main()
{

    snn::SIMDVector a([](size_t x)-> number{ return x;},128);
    snn::SIMDVector b([](size_t x)-> number{ return x;},128);

    std::shared_ptr<snn::NormalizedGaussInit> norm_gauss=std::make_shared<snn::NormalizedGaussInit>(0.f,0.01f);
    std::shared_ptr<snn::GaussInit> gauss=std::make_shared<snn::GaussInit>(0.f,0.1f);

    snn::FullyConnectedNetwork network(1024,gauss);   

    network[14].on_fire=callback;
    network[15].on_fire=callback;

    snn::NumberEncoder input1(0,2,10,1.0,0.0);

    snn::NumberEncoder input2(1,2,10,1.0,0.0);

    input1.setValue(0.0);

    input2.setValue(0.3);

    snn::SIMDVector inputs(0,4);

    snn::SIMDVector input_weight;

    gauss->init(input_weight,4);

    std::cout<<"Network started"<<std::endl;

    std::string plot;

    size_t plot_iter=0;

    for(size_t i=0;i<128;i++)
    {
        plot+="_";
    }

    std::string plot1;

    for(size_t i=0;i<128;i++)
    {
        plot1+="_";
    }

    std::string plot2;

    for(size_t i=0;i<128;i++)
    {
        plot2+="_";
    }

    while(true)
    {
        
        std::cout<<plot<<std::endl;
        std::cout<<plot1<<std::endl<<std::endl;

        std::cout<<plot2<<std::endl;

        input1.update(inputs);

        input2.update(inputs);

        if(inputs[0]>0.5)
        {
            plot[plot_iter]='|';
        }

        if(inputs[1]>0.5)
        {
            plot1[plot_iter]='|';
        }
        
        

        network.excite(inputs);

        snn::SIMDVector output=network.step();

        if(output[14]>0.5)
        {
            plot2[plot_iter]='|';
        }

        plot_iter++;

        std::this_thread::sleep_for(100ms);

        system("clear");

        if( plot_iter > 128 )
        {
            for(size_t i=0;i<128;++i)
            {
                plot[i]='_';
                plot1[i]='_';
                plot2[i]='_';
            }
            plot_iter=0;
        }
    }

    return 0;

   
}
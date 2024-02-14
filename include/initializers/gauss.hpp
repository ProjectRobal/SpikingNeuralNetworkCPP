#pragma once

#include <random>
#include <cmath>

#include "initializer.hpp"

#include "config.hpp"

namespace snn
{
    class GaussInit : public Initializer
    {

        std::normal_distribution<number> gauss;

        public:

        GaussInit(number mean,number std)
        : gauss(mean,std)
        {

        }

        void init(SIMDVector& vec,size_t N)
        {
            std::random_device rd; 

            // Mersenne twister PRNG, initialized with seed from previous random device instance
            std::mt19937 gen(rd()); 

            for(size_t i=0;i<N;++i)
            {
                vec.append(abs(this->gauss(gen)));
            }
        }
    };
}
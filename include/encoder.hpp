#pragma once

#include "simd_vector.hpp"

#include "config.hpp"

/*

A class that will generate stream of impulses based on input

*/


namespace snn
{

    class Encoder
    {
        
        public:

        // update inputs based on it's current state
        virtual void update(SIMDVector& inputs)=0;

    };

}
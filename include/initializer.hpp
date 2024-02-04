#pragma once

#include <simd_vector.hpp>

namespace snn
{
    
    class Initializer
    {
        public:

        virtual void init(SIMDVector& vec,size_t N)=0;
    };

}
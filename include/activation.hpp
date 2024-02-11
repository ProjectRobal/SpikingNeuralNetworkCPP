#pragma once

#include "simd_vector.hpp"

namespace snn
{

    class Activation
    {
        public:

        virtual void activate(SIMDVector& vec)=0;
    };

}
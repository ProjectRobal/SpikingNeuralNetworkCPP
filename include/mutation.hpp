#pragma once

#include "simd_vector.hpp"

namespace snn
{

    class Mutation
    {
        public:

        virtual void mutate(SIMDVector& vec)=0;
    };

}
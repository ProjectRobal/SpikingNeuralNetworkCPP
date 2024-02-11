#pragma once

#include "activation.hpp"

namespace snn
{
    class ReLu : public Activation
    {
        public:

        inline void activate(SIMDVector& vec)
        {
            SIMDVector filtered=vec>0;

            vec=vec*filtered;
        }
    };
}
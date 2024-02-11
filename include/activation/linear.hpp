#pragma once

#include "activation.hpp"

namespace snn
{
    class Linear : public Activation
    {
        public:

        void activate(SIMDVector& vec)
        {
            // do nothing, placeholder activation
        }
    };
}
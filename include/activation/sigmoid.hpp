#pragma once

#include "activation.hpp"

namespace snn
{
    class Sigmoid : public Activation
    {
        SIMDVector power(const SIMDVector& vec, size_t N)
        {
            SIMDVector out=vec;

            while(--N)
            {
                out=out*vec;
            }

            return out;
        }

        SIMDVector exp(const SIMDVector& vec)
        {
            size_t n=1;

            SIMDVector x1=vec;

            SIMDVector sum=x1+1;

            while(n < 20)
            {
                x1=x1*(vec/(++n));
                sum+=x1;
            };

            return sum;
            
        }

        public:

        inline void activate(SIMDVector& vec)
        {
            SIMDVector v=exp(vec);

            vec=v/(v+1);
        }
    };
}
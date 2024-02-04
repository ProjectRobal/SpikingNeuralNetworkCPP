#pragma once

#include "crossover.hpp"
#include "simd_vector.hpp"

namespace snn
{
    class OnePoint : public Crossover
    {
        public:

        SIMDVector cross(const SIMDVector& a,const SIMDVector& b)
        {
            SIMDVector output;

            size_t size=std::min(a.block_count(),b.block_count());

            size_t i=0;

            for(;i<size/2;++i)
            {
                output.append(a.get_block(i));
            }

            for(;i<size;++i)
            {
                output.append(b.get_block(i));
            }

            return output;
        }
    };
}
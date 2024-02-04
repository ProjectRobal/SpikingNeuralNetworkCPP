#pragma once

#include "crossover.hpp"
#include "simd_vector.hpp"

namespace snn
{
    class OnePoint : public Crossover
    {
        public:

        OnePoint()
        {
            
        }


        SIMDVector cross(const SIMDVector& a,const SIMDVector& b)
        {
            SIMDVector output;

            size_t elem_count=std::min(a.size(),b.size());

            size_t size=std::min(a.block_count(),b.block_count());

            size_t i=0;
            
            for(;i<size/2;++i)
            {
                output.append(a.get_block(i));
            }

            // do something with elem_count%32 != 0

            for(;i<=size;++i)
            {
                output.append(b.get_block(i));
            }

            return output;
        }
    };
}
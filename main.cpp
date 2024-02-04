#include <experimental/simd>
#include <iostream>
#include <string_view>

#include "config.hpp"

#include "simd_vector.hpp"

 
 
int main()
{

    snn::SIMDVector a({1,2,3});
    snn::SIMDVector b;
    
    for(size_t i=1;i<=4096;++i)
    {
        a.append(i);
    }

    for(size_t i=1;i<=4096;++i)
    {
        b.append(4096 - i);
    }

    for(int i=0;i<MAX_SIMD_VECTOR_SIZE+3;++i)
    {
        a.pop();
    }

    snn::SIMDVector c=a*b;
    
    std::cout<<a;
   
}
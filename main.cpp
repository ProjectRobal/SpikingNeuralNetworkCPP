#include <experimental/simd>
#include <iostream>
#include <string_view>

#include "config.hpp"

#include "simd_vector.hpp"

#include "neurons/feedforwardneuron.hpp"

#include "crossovers/onepoint.hpp"
#include "crossovers/fastuniform.hpp"
#include "crossovers/fastonepoint.hpp"
 
 
int main()
{

    snn::SIMDVector a;
    snn::SIMDVector b;

    snn::OnePoint cross;
    snn::FastUniform cross1(4096);
    snn::FastOnePoint cross2(4096);
    
    for(size_t i=1;i<=128;++i)
    {
        a.append(i);
    }

    for(size_t i=1;i<=128;++i)
    {
        b.append(i*2);
    }

    snn::SIMDVector c=a*b;
    
    std::cout<<cross2(a,b);
   
}
#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <initializer_list>
#include <functional>

#include "config.hpp"

namespace snn
{

    class SIMDVector
    {

        protected:

        size_t ptr;
        std::vector<SIMD> vec;

        public:

        SIMDVector();

        SIMDVector(std::function<number(size_t)> init_func,size_t N);

        SIMDVector(const std::initializer_list<number>& arr);

        SIMDVector(const SIMDVector& vec);

        SIMDVector(SIMDVector&& vec);

        void set(const number& n, const size_t& i);

        number get(const size_t& i) const;

        number pop();

        number append(number n);

        void append(const SIMD& simd);

        const SIMD& get_block(const size_t& i)
        {
            if(i>=this->vec.size())
            {
                return this->vec.front();
            }

            return this->vec[i];
        }

        SIMDVector operator+(const SIMDVector& v);

        SIMDVector operator-(const SIMDVector& v);

        SIMDVector operator*(const SIMDVector& v);

        SIMDVector operator/(const SIMDVector& v);

        SIMDVector operator*(const number& v);

        void operator+=(const SIMDVector& v);

        void operator-=(const SIMDVector& v);

        void operator*=(const SIMDVector& v);

        void operator/=(const SIMDVector& v);

        void operator*=(const number& v);

        size_t size() const
        {
            return (this->vec.size()-1)*MAX_SIMD_VECTOR_SIZE + ( this->ptr );
        }

        number dot_product();

        number operator[](const size_t& i) const;

        void print(std::ostream& out) const
        {
            for(size_t i=0;i<this->size();++i)
            {
                out<<(*this)[i]<<" ";
            }
        }

        ~SIMDVector();

    };


}

std::ostream& operator<<(std::ostream& out,const snn::SIMDVector& vec);
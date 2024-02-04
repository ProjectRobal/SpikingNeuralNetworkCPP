#include "simd_vector.hpp"


namespace snn
{
    SIMDVector::SIMDVector()
    {
        this->ptr=0;
    }

    SIMDVector::SIMDVector(std::function<number(size_t)> init_func,size_t N)
    : SIMDVector()
    {
        for(size_t i=0;i<N;++i)
        {
            this->append(init_func(i));
        }
    }

    SIMDVector::SIMDVector(const std::initializer_list<number>& arr)
    : SIMDVector()
    {
        for(const auto& a : arr)
        {
            this->append(a);
        }
    }

    SIMDVector::SIMDVector(const SIMDVector& vec)
    {
        this->ptr=vec.ptr;
        std::copy(vec.vec.begin(),vec.vec.end(), std::back_inserter(this->vec));
    }

    SIMDVector::SIMDVector(SIMDVector&& vec)
    {
        this->ptr=vec.ptr;
        vec.ptr=0;

        this->vec=std::move(vec.vec);
        
    }

    void SIMDVector::set(const number& n, const size_t& i)
    {
        if(i>=this->size())
        {
            return;
        }

        this->vec[i/MAX_SIMD_VECTOR_SIZE][i%MAX_SIMD_VECTOR_SIZE]=n;
    }

    number SIMDVector::get(const size_t& i) const
    {
        if(i>=this->size())
        {
            return 0;
        }

        return this->vec[i/MAX_SIMD_VECTOR_SIZE][i%MAX_SIMD_VECTOR_SIZE];
    }

    number SIMDVector::pop()
    {
        number ret=0;

        if(this->ptr==0)
        {
            ret=this->vec.back()[0];
            this->vec.pop_back();
            this->ptr=MAX_SIMD_VECTOR_SIZE-1;

            return ret;
        }

        this->ptr--;

        ret=this->vec.back()[this->ptr];
        this->vec.back()[this->ptr]=0;

        return ret;
    }

    number SIMDVector::append(number n)
    {

        if((this->ptr==MAX_SIMD_VECTOR_SIZE)||(this->vec.empty()))
        {
            this->append(SIMD());            
            this->ptr=0;
        }

        this->vec.back()[this->ptr]=n;
        this->ptr++;

        return n;
    }

    void SIMDVector::append(const SIMD& simd)
    {
        this->vec.push_back(simd);
    }

    SIMDVector SIMDVector::operator+(const SIMDVector& v)
    {
        SIMDVector sv;

        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            sv.append(this->vec[i]+v.vec[i]);
        }

        return sv;
    }

    SIMDVector SIMDVector::operator-(const SIMDVector& v)
    {
        SIMDVector sv;

        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            sv.append(this->vec[i]-v.vec[i]);
        }

        return sv;
    }

    SIMDVector SIMDVector::operator*(const SIMDVector& v)
    {
        SIMDVector sv;

        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            sv.append(this->vec[i]*v.vec[i]);
        }

        return sv;
    }

    SIMDVector SIMDVector::operator/(const SIMDVector& v)
    {
        SIMDVector sv;

        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            sv.append(this->vec[i]/v.vec[i]);
        }

        return sv;
    }

    SIMDVector SIMDVector::operator*(const number& v)
    {
        SIMDVector sv;

        for(size_t i=0;i<this->vec.size();++i)
        {
            sv.append(this->vec[i]*v);
        }

        return sv;   
    }

    void SIMDVector::operator+=(const SIMDVector& v)
    {
        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            this->vec[i]=this->vec[i]+v.vec[i];
        }
    }

    void SIMDVector::operator-=(const SIMDVector& v)
    {
        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            this->vec[i]=this->vec[i]-v.vec[i];
        }
    }

    void SIMDVector::operator*=(const SIMDVector& v)
    {
        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            this->vec[i]=this->vec[i]*v.vec[i];
        }
    }

    void SIMDVector::operator/=(const SIMDVector& v)
    {
        for(size_t i=0;i<std::min(this->vec.size(),v.vec.size());++i)
        {
            this->vec[i]=this->vec[i]/v.vec[i];
        }
    }

    void SIMDVector::operator*=(const number& v)
    {
        for(size_t i=0;i<this->vec.size();++i)
        {
            this->vec[i]=this->vec[i]*v;
        }
    }

    number SIMDVector::dot_product()
    {
        number output=0;

        for(const auto& s : this->vec)
        {
            output+=std::experimental::reduce(s);
        }

        return output;
    }

    number SIMDVector::operator[](const size_t& i) const
    {
        return get(i);
    }

    SIMDVector::~SIMDVector()
    {
        this->vec.clear();
    }
}


std::ostream& operator<<(std::ostream& out,const snn::SIMDVector& vec)
{

    vec.print(out);

    return out;
}
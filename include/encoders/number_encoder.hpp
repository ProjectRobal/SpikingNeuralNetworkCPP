#pragma once

#include "simd_vector.hpp"

#include "config.hpp"

/*

    A simple impulse train number encoder.


*/


namespace snn
{

    class NumberEncoder
    {
        protected:

        number value;

        size_t index;
        size_t ticks;
        size_t ticksToFire;

        size_t min_ticks;
        size_t max_ticks;

        number max_value;
        number min_value;
        
        public:

        NumberEncoder(size_t i,size_t min_ticks,size_t max_ticks,number max_value,number min_value)
        {
            this->min_ticks=min_ticks;
            this->max_ticks=max_ticks;
            this->max_value=max_value;
            this->min_value=min_value;
            this->index=i;
        }

        void setValue(number value)
        {
            this->ticksToFire = max_ticks - ( min_ticks + ((value-min_value)/(max_value-min_value))*max_ticks );

            this->value = value;
        }

        const number& getValue()
        {
            return this->value;
        }

        // update inputs based on it's current state
        void update(SIMDVector& inputs)
        {

            if(this->ticks>=this->ticksToFire)
            {

                inputs.set(1,this->index);

                this->ticks=0;

                return;
            }
            
            inputs.set(0,this->index);
            this->ticks++;
        }


    };

}
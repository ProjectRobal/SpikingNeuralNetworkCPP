#pragma once

#include <iostream>
#include <memory>

#include "simd_vector.hpp"
#include "initializer.hpp"
#include "crossover.hpp"

#include "config.hpp"

namespace snn
{

    class Neuron
    {
        protected:

        long double score;

        public:

        virtual std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)=0;

        virtual void setup(std::shared_ptr<Initializer> init)=0;

        virtual SIMDVector fire(const SIMDVector& input)=0;

        virtual size_t input_size()=0;

        virtual size_t output_size()=0;

        virtual void giveReward(const long double& score)
        {
            this->score+=score;
        }

        virtual void reset()
        {
            this->score=0;
        }

        virtual bool operator < (const Neuron& neuron)
        {
            return this->score < neuron.score;
        }

        virtual bool operator > (const Neuron& neuron)
        {
            return this->score > neuron.score;
        }

        virtual bool operator <= (const Neuron& neuron)
        {
            return this->score <= neuron.score;
        }

        virtual bool operator >= (const Neuron& neuron)
        {
            return this->score >= neuron.score;
        }

        virtual bool operator == (const Neuron& neuron)
        {
            return this->score == neuron.score;
        }

        virtual bool operator != (const Neuron& neuron)
        {
            return this->score != neuron.score;
        }

        virtual ~Neuron(){}
    };

}
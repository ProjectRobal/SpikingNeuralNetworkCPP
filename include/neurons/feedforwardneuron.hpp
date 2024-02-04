#pragma once

#include <iostream>
#include <memory>

#include "simd_vector.hpp"
#include "initializer.hpp"
#include "crossover.hpp"

#include "neuron.hpp"

#include "config.hpp"

namespace snn
{
    template<size_t Input,size_t Output>
    class FeedForwardNeuron : public Neuron
    {
        protected:

        long double score;

        SIMDVector input_weights;
        SIMDVector output_weights;

        public:
        FeedForwardNeuron()
        {

        }

        std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)
        {
            const FeedForwardNeuron<Input,Output>& forward=dynamic_cast<const FeedForwardNeuron<Input,Output>&>(neuron);

            std::shared_ptr<FeedForwardNeuron<Input,Output>> output=std::make_shared<FeedForwardNeuron<Input,Output>>();

            output->input_weights=cross->cross(this->input_weights,forward.input_weights);
            output->output_weights=cross->cross(this->output_weights,forward.output_weights);

            return output;
        }

        void setup(std::shared_ptr<Initializer> init)
        {
            this->input_weights.clear();
            init->init(this->input_weights,Input);

            this->output_weights.clear();
            init->init(this->output_weights,Output);
        }

        SIMDVector fire(const SIMDVector& input)
        {
            number store=(input_weights*input).dot_product();

            return output_weights*store;
        }

        size_t input_size()
        {
            return Input;
        }

        size_t output_size()
        {
            return Output;
        }


    };
}
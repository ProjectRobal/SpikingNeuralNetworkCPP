#pragma once

#include <vector>
#include <memory>
#include <exception>
#include <functional>

#include "simd_vector.hpp"
#include "initializer.hpp"

#include "config.hpp"

namespace snn
{

    class Neuron
    {
        public:
        // a neuron weights
        SIMDVector weights;
        // a callback that activate when neuron fire
        std::function<void(size_t)> on_fire;
        // when set to true it means it cannot be moved
        bool fixed;
    };

    class Network
    {
        protected:

        // it holds each neuron charge

        SIMDVector charges;

        // each neuron weights

        std::vector<Neuron> neurons;

        public:

        Network();

        Network(size_t neuron_size,std::shared_ptr<Initializer> init);

        void setup(size_t neuron_size,std::shared_ptr<Initializer> init);

        // excite a neuron at specified index
        void excite(size_t i);

        // check state of each neuron and execute them 
        void step(); 

        SIMDVector& operator[](size_t i)
        {
            if(i>=this->neurons.size())
            {
                throw std::out_of_range("Neuron out of range!!");
            }

            return this->neurons[i].weights;
        }

        void remove(size_t i)
        {
            if(i>=this->neurons.size())
            {
                throw std::out_of_range("Neuron out of range!!");
            }

            // add a function to remove element at index from SIMD Vector

            this->neurons.erase(this->neurons.begin()+i);

            for(auto& w : this->neurons)
            {
                w.weights.remove(i);
            }

            this->charges.remove(i);
        }

    };

}
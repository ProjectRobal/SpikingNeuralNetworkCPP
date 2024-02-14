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
    };

    class FullyConnectedNetwork
    {
        protected:

        // it holds each neuron charge

        SIMDVector charges;

        // each neuron weights

        std::vector<Neuron> neurons;

        public:

        FullyConnectedNetwork();

        FullyConnectedNetwork(size_t neuron_size,std::shared_ptr<Initializer> init);

        void setup(size_t neuron_size,std::shared_ptr<Initializer> init);

        // excite a neuron at specified index
        void excite(const SIMDVector& inputs);

        // check state of each neuron and execute them 
        SIMDVector step(); 

        Neuron& operator[](size_t i)
        {
            if(i>=this->neurons.size())
            {
                throw std::out_of_range("Neuron out of range!!");
            }

            return this->neurons[i];
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

        void insert(size_t i,std::shared_ptr<Initializer> init)
        {
            Neuron new_neuron;

            init->init(new_neuron.weights,this->size()+1);

            this->charges.insert(i,0);

            this->neurons.insert(this->neurons.begin()+i,new_neuron);
        }

        size_t size()
        {
            return this->neurons.size();
        }

    };

}
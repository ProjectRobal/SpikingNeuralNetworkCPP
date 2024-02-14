#include "Network.hpp"

namespace snn
{
    Network::Network()
    {

    }

    Network::Network(size_t neuron_size,std::shared_ptr<Initializer> init)
    {
        this->setup(neuron_size,init);
    }

    void Network::setup(size_t neuron_size,std::shared_ptr<Initializer> init)
    {
        for(size_t i=0;i<neuron_size;++i)
        {
            Neuron neuron;

            init->init(neuron.weights,neuron_size);

            this->neurons.push_back(neuron);
        }

        this->charges=SIMDVector(0,neuron_size);

    }

    void Network::excite(size_t i)
    {
        if(i>=this->neurons.size())
        {
            throw std::out_of_range("Neuron out of range!!");
        }

        this->charges+=this->neurons[i].weights;
    }

    void Network::step()
    {
        SIMDVector check=this->charges<NEURON_THRESHOLD_LEVEL;

        // clear fired neurons
        this->charges=this->charges*check;

        for(size_t i=0;i<check.size();++i)
        {
            if(check[i]<1)
            {
                this->charges+=this->neurons[i].weights;
            }
        }

    }
}
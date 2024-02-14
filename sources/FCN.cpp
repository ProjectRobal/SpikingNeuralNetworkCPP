#include "FCN.hpp"
#include <iostream>


namespace snn
{
    FullyConnectedNetwork::FullyConnectedNetwork()
    {

    }

    FullyConnectedNetwork::FullyConnectedNetwork(size_t neuron_size,std::shared_ptr<Initializer> init)
    {
        this->setup(neuron_size,init);
    }

    void FullyConnectedNetwork::setup(size_t neuron_size,std::shared_ptr<Initializer> init)
    {
        for(size_t i=0;i<neuron_size;++i)
        {
            Neuron neuron;

            init->init(neuron.weights,neuron_size);

            this->neurons.push_back(neuron);
        }

        this->charges=SIMDVector(0,neuron_size);

    }

    void FullyConnectedNetwork::excite(const SIMDVector& inputs)
    {
        this->charges+=inputs;
    }

    SIMDVector FullyConnectedNetwork::step()
    {
        SIMDVector check=this->charges<NEURON_THRESHOLD_LEVEL;

        this->charges*=check;

        for(size_t i=0;i<check.size();++i)
        {
            if(check[i]<0.5)
            {
                number my_own_charge=this->charges[i];

                this->charges+=this->neurons[i].weights;

                this->charges.set(my_own_charge,i);

                if(this->neurons[i].on_fire)
                {
                    this->neurons[i].on_fire(i);
                }

            }
        }

        return check<0.5;

    }
}
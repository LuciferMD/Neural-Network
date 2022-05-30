using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neural_Network
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        List<Layer> Layers { get; }

        public NeuralNetwork(Topology Topology)
        {
            this.Topology = Topology;

            this.Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(List<double> inputSignals) //inputSiganls = Topology.InputCount !!!
        {
            SendSignalsToInputNeurons(inputSignals); //send first data to network
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers[^1].Neurons[0];
            }
            else
            {
                return Layers[^1].Neurons.OrderByDescending(n => n.Output).First(); //choose the most probably
            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++) //from 1 !! because we already sent data on first layer
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].getSignals();

                foreach (var neuron in layer.Neurons) //we calculate signal on previous level and send to the next level
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()  //Create first level
        {
            List<Neuron> inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)   // number of neurons in Topology.inputCount
            {
                Neuron neuron = new Neuron(1, NeuronType.Input); //create neurons for first level with NeuronType = input
                inputNeurons.Add(neuron);
            }

            Layer InputLayer = new Layer(inputNeurons, NeuronType.Input);
            this.Layers.Add(InputLayer);                                    // add our first lvl
        }
        private void CreateOutputLayer() //Create and add last layer
        {
            List<Neuron> outputNeurons = new List<Neuron>();
            var lastLayer = Layers[^1]; //[integerList.Count - 1] - last

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                Neuron neuron = new Neuron(lastLayer.Count, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            Layer OutputLayer = new Layer(outputNeurons, NeuronType.Output);

            this.Layers.Add(OutputLayer);
        }
        private void CreateHiddenLayers() //Create and add hidden layers
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers[^1]; //[integerList.Count - 1] - last

                for (int i = 0; i < Topology.HiddenLayers[j]; i++) //number neurons in hidden j layer
                {
                    Neuron neuron = new Neuron(lastLayer.Count);
                    hiddenNeurons.Add(neuron);
                }

                Layer hiddenLayer = new Layer(hiddenNeurons);

                this.Layers.Add(hiddenLayer);
            }
        }


    }
}

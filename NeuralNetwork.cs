using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neural_Network
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology Topology)
        {
            this.Topology = Topology;

            this.Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params double[] inputSignals) //inputSiganls = Topology.InputCount !!!
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

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
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
                Neuron neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
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
                    Neuron neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }

                Layer hiddenLayer = new Layer(hiddenNeurons);

                this.Layers.Add(hiddenLayer);
            }
        }

        public double Learn(List<Tuple<double,double []>> dataset, int epoch) //epoch - number of passing BackPropagation for all network
        {
            double error = 0;
            for (int i = 0; i < epoch; i++)
            {
                foreach (var data in dataset)
                {
                    error += BackPropagation(data.Item1, data.Item2); //data.Item1 - expected //data.Item2 //double[] inputs
                }
            }

            var result = error / epoch; //average mistake 
            return result;
        }
        private double BackPropagation(double expected, params double[] inputs) // revers propagation of the error
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers[^1].Neurons) //for last lvl
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int i = Layers.Count-2; i >=0 ; i--)  //-1 - we start from 0//-2 we already teach last layer
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];

                for (int j = 0; j < layer.NeuronCount; j++) //for check all neurons on current level
                {
                    var neuron = layer.Neurons[i]; 

                    for (int k = 0; k < previousLayer.NeuronCount; k++)   //number of input == number of output on previos level
                    {
                        var previousNeuron = previousLayer.Neurons[k];

                        var error = previousNeuron.Weights[i] * previousNeuron.Delta; // i = current neuron
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return difference * difference; // to avoid negative number and increase error
        }


    }
}

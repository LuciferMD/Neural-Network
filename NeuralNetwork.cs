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

        public Neuron Predict(params double[] inputSignals) //inputSiganls = Topology.InputCount !!!
        {
           /// var signals = Normalization(inputSignals);

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

        public double Learn(double[] expected, double[,] inputs, int epoch) //epoch - number of passing BackPropagation for all network
        {
            var signals = Normalization(inputs);

            double error = 0;

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(signals, j);

                    error += BackPropagation(output, input); 

                }
            }


            var result = error / epoch; //average mistake 
            return result;
        }
        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row, i];
            return array;
        }
        private double[,] Scaling(double[,] inputs) //Масштабирование
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];
                for (int row = 1; row < inputs.GetLength(0); row++) //calculate min and max number for scaling for each column
                {
                    var item = inputs[row, column];
                    if (item < min)
                    {
                        min = item;
                    }

                    if (item > max)
                    {
                        max = item;
                    }
                }
                var devider = (max - min);
                for (int row = 0; row < inputs.GetLength(0); row++) // and scaling each value by formula
                {
                    result[row, column] = (inputs[row, column] - min) / devider;
                }
            }
            return result;
        }
        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                double sum = 0;

                //calculate average number of signal
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                //standard square deviation neuron
                double error = 0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column]), 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                //output element
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }
            return result;
        }


        private double BackPropagation(double expected, params double[] inputs) // revers propagation of the error
        {
            var actual = Predict(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers[^1].Neurons) //for last lvl
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int i = Layers.Count - 2; i >= 0; i--)  //-1 - we start from 0//-2 we already teach last layer
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

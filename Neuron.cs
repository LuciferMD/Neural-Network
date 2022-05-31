
using System;
using System.Collections.Generic;

namespace Neural_Network
{
    public class Neuron
    {
        public List<double> Weights { get; } //the importance of neuron(communication)
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }
        public Neuron(int? inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeightRandom(inputCount);

        }

        private void InitWeightRandom(int? inputCount)
        {
            var rand = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if(NeuronType== NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rand.NextDouble());
                }
                Inputs.Add(0);
            }
        }   

        public void Learn(double error, double LearningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);  //error = actual result - expect resut

            for (int i = 0; i < Weights.Count; i++) //for all INPUT
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * LearningRate; //calculate new weight
                Weights[i] = newWeight;
            } 
            
        }
        public double FeedForward(List<double> inputs) //feed forward neural networks  (FF)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }
            double sum = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else //for 1 lvl weight always = 1 or 0
            {
                Output = sum;
            }
            return Output;
        }
        private double Sigmoid(double x)
        {
            double result = 1.0 / (1.0 + Math.Pow(Math.E, -x)); //to "smooth out" the values of a certain value.
            return result;
        }
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }
        public override string ToString() //For Debuging
        {
            return Output.ToString();
        }
    }
}

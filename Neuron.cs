﻿
using System;
using System.Collections.Generic;

namespace Neural_Network
{
    public class Neuron
    {
        public List<double> Weights { get; } //the importance of neuron(communication)
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }

        public Neuron(int? inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weights = new List<double>();

            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }

        }

        public void SetWights(params double[] weights) //delete
        {
            for (int i = 0; i < weights.Length; i++)
            {
                this.Weights[i] = weights[i];
            }
        }
        public double FeedForward(List<double> inputs) //feed forward neural networks  (FF)
        {
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
        public override string ToString() //For Debuging
        {
            return Output.ToString();
        }
    }
}

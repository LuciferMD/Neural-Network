using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_Network
{
    public class Topology
    {
        public int? InputCount { get; } //Input neuron on first layer
        public int? OutputCount { get; } //Output neuron on last layer
        public double LearningRate { get; }
        public List<int> HiddenLayers { get; }// Count hidden neurons

        public Topology(int InputCount, int OutputCount,double LearningRate, params int[] HiddenLayers)
        {
            this.InputCount = InputCount;
            this.OutputCount = OutputCount;
            this.LearningRate = LearningRate;

            this.HiddenLayers = new List<int>();
            this.HiddenLayers.AddRange(HiddenLayers);

        }
    }
}

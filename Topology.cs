using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_Network
{
    public class Topology
    {
        public int? InputCount { get; } //Input neuron on first layer
        public int? OutputCount { get; } //Output neuron on last layer
        public List<int> HiddenLayers { get; }// Count hidden neurons

        public Topology(int InputCount, int OutputCount, params int[] HiddenLayers)
        {
            this.InputCount = InputCount;
            this.OutputCount = OutputCount;

            this.HiddenLayers = new List<int>();
            this.HiddenLayers.AddRange(HiddenLayers);

        }
    }
}

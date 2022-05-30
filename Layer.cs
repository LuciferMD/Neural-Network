using System.Collections.Generic;

namespace Neural_Network
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int? Count => Neurons?.Count ?? 0;

        public Layer(List<Neuron> Neurons, NeuronType type = NeuronType.Normal) // One level - one type
        {
            this.Neurons = Neurons;
        }

        public List<double> getSignals() //collect all signals from layer
        {
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }
    }
}

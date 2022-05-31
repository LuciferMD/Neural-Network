using System.Collections.Generic;

namespace Neural_Network
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int? NeuronCount => Neurons?.Count ?? 0;

        public NeuronType Type { get; }
        public Layer(List<Neuron> Neurons, NeuronType Type = NeuronType.Normal) // One level - one type
        {
            this.Neurons = Neurons;
            this.Type = Type;
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

        public override string ToString() //for debug
        {
            return Type.ToString(); 
        }
    }
}

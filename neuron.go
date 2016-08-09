package neugo

type Neuron struct {
	weights    []float64
	activation ActivationFunc
}

func NewNeuron(c *Config) *Neuron {
	return &Neuron{
		weights:    make([]float64, c.NumWeights),
		activation: c.Activation,
	}
}

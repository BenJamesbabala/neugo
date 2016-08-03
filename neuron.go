package neugo

type Neuron struct {
	weights []float64
}

func NewNeuron() *Neuron {
	return &Neuron{}
}

package neugo

import (
	"log"
	"math"
)

const (
	BIAS = -1.0 // bias for activation
)

type NeuralNet struct {
	conf    *Config   // neural net configuration
	weights []float64 // weights for activation
}

// Create a new neural network with 0 weights.
func NewNeuralNet(conf *Config) *NeuralNet {
	return &NeuralNet{
		conf: conf,
		weights: func(c *Config) []float64 {
			// number of weights including bias
			numWeights := (c.NumInputs+1)*c.NumNeurons +
				(c.NumLayers-1)*(c.NumNeurons+1)*c.NumNeurons +
				(c.NumNeurons+1)*c.NumOutputs
			weights := make([]float64, numWeights)
			return weights
		}(conf),
	}
}

// Get the neural network's weights.
func (n *NeuralNet) Weights() []float64 {
	return n.weights
}

// Get the number of neural network's weights.
func (n *NeuralNet) NumWeights() int {
	return len(n.weights)
}

// Build weights by decoding a DNA and
// generate neural network's weights.
func (n *NeuralNet) Build(d *DNA) {
	if d.Size()%len(n.weights) != 0 {
		log.Fatal("Invalid DNA size")
	}
	weightLen := d.Size() / len(n.weights)
	for i, _ := range n.weights {
		weight := d[i*weightLen : (i+1)*weightLen]
		decoded := func() float64 {
			max := math.Pow(2.0, len(weight)) - 1

		}
	}
}

// Update the neural network and return output
// given a set of inputs.
func (n *NeuralNet) Update(inputs []float64) []float64 {
	if len(inputs) != n.conf.NumInputs {
		log.Fatal("Invalid inputs")
	}
	return n.update(inputs, 0)
}

// recursive neural network update helper function
func (n *NeuralNet) update(inputs []float64, counter int) []float64 {
	// hidden layer -> output layer
	last := n.NumWeights() - (n.conf.NumNeurons+1)*n.conf.NumOutputs
	if counter == last {
		outputs := make([]float64, n.conf.NumOutputs)
		for i, _ := range outputs {
			for _, val := range inputs {
				outputs[i] += val * n.weights[counter]
				counter++
			}
			// add bias
			outputs[i] += n.weights[counter] * BIAS
			counter++
		}
		// apply activation function
		for i, _ := range outputs {
			outputs[i] = n.conf.Activation(outputs[i])
		}
		//fmt.Printf("progress: %f\n", outputs)
		return outputs
	}
	// input -> hidden layer -> hidden layer
	outputs := make([]float64, n.conf.NumNeurons)
	for i, _ := range outputs {
		for _, val := range inputs {
			outputs[i] += val * n.weights[counter]
			counter++
		}
		// add bias
		outputs[i] += n.weights[counter] * BIAS
		counter++
	}
	// apply activation function
	for i, _ := range outputs {
		outputs[i] = n.conf.Activation(outputs[i])
	}
	//fmt.Printf("progress: %f\n", outputs)
	return n.update(outputs, counter)
}

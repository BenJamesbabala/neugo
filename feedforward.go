package neugo

import "errors"

var (
	ErrInputs = errors.New("invalid inputs")
)

const (
	BIAS = -1.0 // bias for activation
)

type FeedForward struct {
	conf    *Config   // neural net configuration
	weights []float64 // weights for activation
}

// Create a new neural network with 0 weights.
func NewFeedForward(conf *Config) *FeedForward {
	return &FeedForward{
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
func (f *FeedForward) Weights() []float64 {
	return f.weights
}

// Get the number of neural network's weights.
func (f *FeedForward) NumWeights() int {
	return len(f.weights)
}

// Update the neural network and return output
// given a set of inputs.
func (f *FeedForward) Update(inputs []float64) ([]float64, error) {
	if len(inputs) != f.conf.NumInputs {
		return nil, ErrInputs
	}
	return f.update(inputs, 0), nil
}

// recursive neural network update helper function
func (f *FeedForward) update(inputs []float64, counter int) []float64 {
	// hidden layer -> output layer
	last := f.NumWeights() - (f.conf.NumNeurons+1)*f.conf.NumOutputs
	if counter == last {
		outputs := make([]float64, f.conf.NumOutputs)
		for i, _ := range outputs {
			for _, val := range inputs {
				outputs[i] += val * f.weights[counter]
				counter++
			}
			// add bias
			outputs[i] += f.weights[counter] * BIAS
			counter++
		}
		// apply activation function
		for i, _ := range outputs {
			outputs[i] = f.conf.Activation(outputs[i])
		}
		//fmt.Printf("progress: %f\n", outputs)
		return outputs
	}
	// input -> hidden layer -> hidden layer
	outputs := make([]float64, f.conf.NumNeurons)
	for i, _ := range outputs {
		for _, val := range inputs {
			outputs[i] += val * f.weights[counter]
			counter++
		}
		// add bias
		outputs[i] += f.weights[counter] * BIAS
		counter++
	}
	// apply activation function
	for i, _ := range outputs {
		outputs[i] = f.conf.Activation(outputs[i])
	}
	//fmt.Printf("progress: %f\n", outputs)
	return f.update(outputs, counter)
}

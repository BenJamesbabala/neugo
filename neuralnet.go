package neugo

import "errors"

var (
	// wrong number of inputs
	ErrInputLen = errors.New("invalid input length")
)

type NeuralNet struct {
	conf    *Config   // configuration
	weights []*Matrix // list of weights
}

func NewNeuralNet(conf *Config) *NeuralNet {
	return &NeuralNet{
		conf: conf,
		weights: func() []*Matrix {
			weights := make([]*Matrix, 2)
			// input layer -> hidden layer
			weights[0] = NewNorm(conf.NumInput,
				conf.NumHidden, 0.0, 6.0)
			// hidden layer[t] -> hidden layer[t+1]
			for i := 1; i < conf.NumLayer-1; i++ {
				weights[i] = NewNorm(conf.NumHidden,
					conf.NumHidden, 0.0, 6.0)
			}
			// hidden layer -> output layer
			weights[conf.NumLayer-1] = NewNorm(conf.NumHidden,
				conf.NumOutput, 0.0, 6.0)
			return weights
		}(),
	}
}

// Get the neural network's weights.
func (n *NeuralNet) Weights() []*Matrix {
	return weights
}

// Activate the neural network.
func (n *NeuralNet) Activate(inputs []float64) []float64 {
	if len(inputs) != n.conf.NumInput {
		return nil, ErrInputLen
	}
}

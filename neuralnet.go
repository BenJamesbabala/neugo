package neugo

import (
	"errors"
	"math"

	"github.com/jinseokYeom/neugo/matrix"
)

const (
	BIAS = -1.0
)

var (
	// wrong number of inputs
	ErrInputLen = errors.New("invalid input length")
)

type NeuralNet struct {
	conf    *Config          // configuration
	weights []*matrix.Matrix // list of weights
	memory  []*matrix.Matrix // memory of activations
}

func NewNeuralNet(conf *Config) (*NeuralNet, error) {
	// generate weights of neural net
	weights := make([]*matrix.Matrix, conf.NumLayer+1)
	// input layer -> hidden layer
	il, err := matrix.NewNorm(
		conf.NumInput+1, // number of inputs + 1 (BIAS)
		conf.NumHidden,  // number of hidden neurons
		0.0,             // mean = 0
		6.0,             // std. dev = 6
	)
	if err != nil {
		return nil, err
	}
	weights[0] = il
	// hidden layer[t] -> hidden layer[t+1]
	for i := 1; i < conf.NumLayer; i++ {
		hl, err := matrix.NewNorm(
			conf.NumHidden+1, // number of hidden neurons + 1 (BIAS)
			conf.NumHidden,   // number of hidden neurons
			0.0,              // mean = 0
			6.0,              // std. dev = 6
		)
		if err != nil {
			return nil, err
		}
		weights[i] = hl
	}
	// hidden layer -> output layer
	ol, err := matrix.NewNorm(
		conf.NumHidden+1, // number of hidden nuerons + 1 (BIAS)
		conf.NumOutput,   // number of outputs
		0.0,              // mean = 0
		6.0,              // std. dev = 6
	)
	if err != nil {
		return nil, err
	}
	weights[conf.NumLayer] = ol
	return &NeuralNet{
		conf:    conf,
		weights: weights,
		memory:  make([]*matrix.Matrix, conf.NumLayer+1),
	}, nil
}

// Get the neural network's weights.
func (n *NeuralNet) Weights() []*matrix.Matrix {
	return n.weights
}

// Activate the neural network and feedforward.
func (n *NeuralNet) Feedforward(inputs []float64) ([]float64, error) {
	if len(inputs) != n.conf.NumInput {
		return nil, ErrInputLen
	}
	for i, w := range n.weights {
		inputs = append(inputs, BIAS)
		im, _ := matrix.New(1, len(inputs), inputs)
		outputs, _ := im.Mult(w)
		// apply activation function
		outputs, _ = outputs.Func(n.conf.Activation)
		// store activation output
		n.memory[i] = outputs
		inputs = outputs.Data()
	}
	return inputs, nil
}

// Backpropagation for training, given a prediction and an actual answer.
func (n *NeuralNet) Backpropagate(pred, actual []float64) {
	err := make([]float64, len(pred))
	for i, _ := range err {
		// cost function
		err[i] = math.Pow(pred[i]-actual[i], 2) / 2.0
	}

}

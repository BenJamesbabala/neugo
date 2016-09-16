package neugo

import (
	"errors"
	"fmt"
	"math"

	"github.com/jinseokYeom/migo"
)

var (
	// wrong number of inputs
	ErrInputLen = errors.New("invalid input length")
	// wrong number of weights
	ErrWeightLen = errors.New("invalid weigth length")
)

type NeuralNet struct {
	conf    *Config        // configuration
	weights []*migo.Matrix // list of weights
	memory  []*migo.Matrix // memory of activations
}

func NewNeuralNet(conf *Config) (*NeuralNet, error) {
	// generate weights of neural net
	weights := make([]*migo.Matrix, conf.NumLayer+1)
	// input layer -> hidden layer
	il, err := migo.NewNorm(
		conf.NumInput+1,   // number of inputs + 1 (BIAS)
		conf.NumHidden,    // number of hidden neurons
		conf.WeightMean,   // mean
		conf.WeightStdDev, // std. dev
	)
	if err != nil {
		return nil, err
	}
	weights[0] = il
	// hidden layer[t] -> hidden layer[t+1]
	for i := 1; i < conf.NumLayer; i++ {
		hl, err := migo.NewNorm(
			conf.NumHidden+1,  // number of hidden neurons + 1 (BIAS)
			conf.NumHidden,    // number of hidden neurons
			conf.WeightMean,   // mean
			conf.WeightStdDev, // std. dev
		)
		if err != nil {
			return nil, err
		}
		weights[i] = hl
	}
	// hidden layer -> output layer
	ol, err := migo.NewNorm(
		conf.NumHidden+1,  // number of hidden nuerons + 1 (BIAS)
		conf.NumOutput,    // number of outputs
		conf.WeightMean,   // mean
		conf.WeightStdDev, // std. dev
	)
	if err != nil {
		return nil, err
	}
	weights[conf.NumLayer] = ol
	return &NeuralNet{
		conf:    conf,
		weights: weights,
		memory:  make([]*migo.Matrix, conf.NumLayer+1),
	}, nil
}

// Get the total number of neural network's weights.
func (n *NeuralNet) NumWeights() int {
	return (n.conf.NumInput+1)*n.conf.NumHidden +
		(n.conf.NumHidden+1)*n.conf.NumHidden*n.conf.NumLayer +
		(n.conf.NumHidden+1)*n.conf.NumOutput
}

// Get the neural network's weights.
func (n *NeuralNet) Weights() []*migo.Matrix {
	return n.weights
}

// Get the neural network's current memory.
func (n *NeuralNet) Memory() []*migo.Matrix {
	return n.memory
}

// Build the neural network given a list of weights; return an
// error if wrong number of weights are provided.
func (n *NeuralNet) Build(weights []float64) error {
	if len(weights) != n.NumWeights() {
		return ErrWeightLen
	}
	// input layer -> hidden layer
	ih := (n.conf.NumInput + 1) * n.conf.NumHidden
	ihWeights := weights[:ih]
	ihMat, err := migo.New(
		n.conf.NumInput+1,
		n.conf.NumHidden,
		ihWeights,
	)
	if err != nil {
		return err
	}
	n.weights[0] = ihMat
	// hidden layer -> hidden layer
	prev := ih
	for i := 1; i < n.conf.NumLayer; i++ {
		next := prev + (n.conf.NumHidden+1)*n.conf.NumHidden
		hhWeights := weights[prev:next]
		hhMat, err := migo.New(
			n.conf.NumHidden+1,
			n.conf.NumHidden,
			hhWeights,
		)
		if err != nil {
			return err
		}
		n.weights[i] = hhMat
		prev = next
	}
	// hidden layer -> output layer
	hoWeights := weights[prev:]
	hoMat, err := migo.New(
		n.conf.NumHidden+1,
		n.conf.NumOutput,
		hoWeights,
	)
	if err != nil {
		return err
	}
	n.weights[len(n.weights)-1] = hoMat
	return nil
}

// Activate the neural network and feedforward. Return an error
// if wrong number of inputs are provided.
func (n *NeuralNet) Feedforward(inputs []float64) ([]float64, error) {
	if len(inputs) != n.conf.NumInput {
		return nil, ErrInputLen
	}
	for i, w := range n.weights {
		inputs = append(inputs, n.conf.Bias)
		im, _ := migo.New(1, len(inputs), inputs)
		outputs, _ := im.Mult(w)
		// apply activation function
		fmt.Printf("OUTPUT NUM ROW: %d\n", outputs.NumRow())
		fmt.Printf("OUTPUT NUM COL: %d\n", outputs.NumColumn())
		fmt.Printf("OUTPUT DATA: %f\n", outputs.Data())
		signal := outputs.Func(n.conf.Activation)
		// store activation output
		n.memory[i].Copy(signal)
		inputs = outputs.Data()
	}
	return inputs, nil
}

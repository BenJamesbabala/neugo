package neugo

import (
	"fmt"
	"math/rand"
	"testing"
)

/*
   Test Deep Neural Network

   Number of inputs:                    3
   Number of outputs:                   2
   Number of hidden layers:             3
   Number of neurons in a hidden layer: 4

   -> I1  H1,1  H2,1  H3,1  O1 ->
   -> I2  H1,2  H2,2  H3,2  O2 ->
   -> I3  H1,3  H2,3  H3,3
          H1,4  H2,4  H3,4
        /     /     /     /
     BIAS  BIAS  BIAS  BIAS

*/
func TestNeuralNet(t *testing.T) {
	conf := &Config{
		NumInputs:  3,
		NumOutputs: 2,
		NumLayers:  3,
		NumNeurons: 4,
	}
	nn := NewFeedForward(conf)

	inputs := make([]float64, conf.NumInputs)
	for i, _ := range inputs {
		inputs[i] = rand.NormFloat64() * 5.0
	}
	outputs := nn.Update(inputs)

	fmt.Printf("Input: %f\n", inputs)
	fmt.Printf("Output: %f\n", outputs)
}

package neugo

import (
	"fmt"
	"log"
	"testing"
)

func TestNeuralNet(t *testing.T) {
	nn, err := NewNeuralNet(&Config{
		NumInput:     3,
		NumOutput:    2,
		NumHidden:    4,
		NumLayer:     3,
		Bias:         -1.0,
		WeightMean:   0.0,
		WeightStdDev: 6.0,
		Activation:   Sigmoid(),
	})
	if err != nil {
		log.Fatal(err)
	}
	// print all the weights
	for i, w := range nn.weights {
		fmt.Printf("LAYER %d\n", i)
		w.Print()
	}
	outputs, err := nn.Feedforward([]float64{0.1, 0.4, 0.63})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("outputs: %f\n", outputs)

	// power test
	for i := 1; i < 100; i++ {
		nn, err = NewNeuralNet(&Config{
			NumInput:     3,
			NumOutput:    2,
			NumHidden:    i,
			NumLayer:     i,
			Bias:         -1.0,
			WeightMean:   0.0,
			WeightStdDev: 6.0,
			Activation:   Sigmoid(),
		})
		if err != nil {
			log.Fatal(err)
		}
		outputs, err := nn.Feedforward([]float64{0.1, 0.4, 0.63})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("outputs: %f\n", outputs)
	}
}

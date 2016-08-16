package neugo

import "math"

// Environment is defined as a function that takes a
// neural network and puts it to test. It can either perform
// reinforcement learning and return the fitness, or perform
// a supervised learning and return the error.
type Environment func(*NeuralNet) float64

// Test environment 1: XOR
// In this environment, a neural network is tested on
// XOR problems: the neural net is tested on 0 xor 0,
// 0 xor 1, 1 xor 0, and 1 xor 1; this environment returns
// total error for all four tests.
func XORTest() Environment {
	return func(nn *NeuralNet) float64 {
		totalErr := 0.0
		// 0 xor 0
		output, _ := nn.Feedforward([]float64{0.0, 0.0})
		totalErr += math.Pow(output[0]-0.0, 2.0)
		output, _ = nn.Feedforward([]float64{0.0, 1.0})
		totalErr += math.Pow(output[0]-1.0, 2.0)
		output, _ = nn.Feedforward([]float64{1.0, 0.0})
		totalErr += math.Pow(output[0]-1.0, 2.0)
		output, _ = nn.Feedforward([]float64{1.0, 1.0})
		totalErr += math.Pow(output[0]-0.0, 2.0)
		return totalErr
	}
}

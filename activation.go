package ne

import "math"

// Activation function takes in a float64 value as an input,
// and returns 0 or 1 as activation signal.
type ActivationFunc func(float64) float64

// Step function returns 1 if input is larger than 0.5,
// and returns 0 otherwise.
func Step() ActivationFunc {
	return func(x float64) float64 {
		if x > 0.5 {
			return 1.0
		}
		return 0.0
	}
}

// Sigmoid function is a S-shaped curve that returns 0 or 1
// as activation signal, given a response rate.
func Sigmoid(resp float64) ActivationFunc {
	return func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x/resp))
	}
}

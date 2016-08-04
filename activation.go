package neugo

import "math"

// Activation function takes in a float64 value as an input,
// and returns 0 or 1 as activation signal.
type ActivationFunc func(float64) float64

// Step function returns 1 if input is positive,
// and returns 0 otherwise.
func Step() ActivationFunc {
	return func(x float64) float64 {
		if x < 0 {
			return 0.0
		}
		return 1.0
	}
}

// Sigmoid function is a S-shaped curve that returns 0 or 1
// as activation signal.
func Sigmoid() ActivationFunc {
	return func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
}

// Tanh function is a S-shaped curve that returns -1 or 1
// as activation signal.
func Tanh() ActivationFunc {
	return func(x float64) float64 {
		return 2/(1.0+math.Exp(-2*x)) - 1
	}
}

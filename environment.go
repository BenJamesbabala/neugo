package neugo

// Environment is defined as a function that takes a
// neural network and puts it to test. It can either perform
// reinforcement learning and return the fitness, or perform
// a supervised learning and return the error.
type Environment func(*NeuralNet) float64

package neugo

type NeuralNet struct {
	conf    *Config   // configuration
	weights []*Matrix // list of weights
}

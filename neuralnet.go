package neugo

type NeuralNet struct {
	conf    *Config   // configuration
	weights []*Matrix // list of weights
}

func NewNeuralNet(conf *Config) *NeuralNet {
	return &NeuralNet{
		conf: conf,
		weights: func() []*Matrix {
			weights := make([]*Matrix, 2)

			// to be implemented

			return weights
		}(),
	}
}

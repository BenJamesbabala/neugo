package neugo

type Config struct {
	NumInput     int            // number of inputs
	NumOutput    int            // number of outputs
	NumHidden    int            // number of hidden neurons in a layer
	NumLayer     int            // number of hidden layers
	Bias         float64        // bias for neural net
	WeightMean   float64        // weight mean
	WeightStdDev float64        // weight standard deviation
	Activation   ActivationFunc // activation function
}

// Create a new configuration set to null.
func NewConfig() *Config {
	return &Config{}
}

// Set number of inputs.
func (c *Config) SetNumInput(n int) {
	c.NumInput = n
}

// Set number of outputs.
func (c *Config) SetNumOutput(n int) {
	c.NumOutput = n
}

// Set number of neurons in hidden layers.
func (c *Config) SetNumHidden(n int) {
	c.NumHidden = n
}

// Set number of hidden layers.
func (c *Config) SetNumLayer(n int) {
	c.NumLayer = n
}

// Set bias for neural net.
func (c *Config) SetBias(n float64) {
	c.Bias = n
}

// Set weight mean for neural net.
func (c *Config) SetWeightMean(n float64) {
	c.WeightMean = n
}

// Set weight standard deviation for neural net.
func (c *Config) SetWeightStdDev(n float64) {
	c.WeightStdDev = n
}

// Set activation function.
func (c *Config) SetActivation(fn ActivationFunc) {
	c.Activation = fn
}

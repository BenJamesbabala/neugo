package neugo

type Config struct {
	NumInput   int            // number of inputs
	NumOutput  int            // number of outputs
	NumHidden  int            // number of hidden neurons in a layer
	NumLayer   int            // number of hidden layers
	Activation ActivationFunc // activation function
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

// Set activation function.
func (c *Config) SetActivation(fn ActivationFunc) {
	c.Activation = fn
}

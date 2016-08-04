package neugo

type Config struct {
	NumInputs  int            // number of inputs
	NumOutputs int            // number of outputs
	NumLayers  int            // number of hidden layers
	NumNeurons int            // number of hidden neurons in a layer
	Activation ActivationFunc // activation function
}

// Create a new configuration set to null.
func NewConfig() *Config {
	return &Config{}
}

// Set number of inputs.
func (c *Config) SetNumInputs(n int) {
	c.NumInputs = n
}

// Set number of outputs.
func (c *Config) SetNumOutputs(n int) {
	c.NumOutputs = n
}

// Set number of layers.
func (c *Config) SetNumLayers(n int) {
	c.NumLayers = n
}

// Set number of neurons in hidden layers.
func (c *Config) SetNumNeurons(n int) {
	c.NumNeurons = n
}

// Set activation function.
func (c *Config) SetActivation(fn ActivationFunc) {
	c.Activation = fn
}

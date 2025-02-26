  def initialize_parameters(self):
        """Initialize weights with He initialization and biases to zero."""
        for i in range(self.num_layers - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b =  lf.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
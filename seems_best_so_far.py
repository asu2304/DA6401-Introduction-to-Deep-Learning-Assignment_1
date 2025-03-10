import numpy as np
from keras.datasets import fashion_mnist
import wandb

class backprop_from_scratch:
    
    def __init__(self, layer_size, activation_function='sigmoid'):
        
        """
        Initialize the neural network with the specified layer sizes and activation function.
        
        Args:
            layer_size (list): List of integers specifying the size of each layer.
            activation_function (str): Activation function for hidden layers ('sigmoid', 'relu', 'tanh', 'linear').
        """
        
        # Validate activation function
        valid_activations = ['sigmoid', 'relu', 'tanh', 'linear']
        if activation_function.lower() not in valid_activations:
            raise ValueError(f"Activation function must be one of {valid_activations}, got {activation_function}")
        self.activation_function = activation_function.lower()
        
        # Initialize network parameters
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_size)
        self.layer_sizes = layer_size
        self.initialize_params()
        
        # Initialize histories and velocities for optimizers
        self.history_weights = [np.zeros_like(w) for w in self.weights]
        self.history_bias = [np.zeros_like(b) for b in self.biases]
        self.weights_velocity = [np.zeros_like(w) for w in self.weights]
        self.bias_velocity = [np.zeros_like(b) for b in self.biases]
        
        # Default beta values for optimizers
        self.beta_1 = 0.900
        self.beta_2 = 0.999

    def initialize_params(self):
        """Initialize weights with He initialization and biases with zeros."""
        for i in range(self.num_layers - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def apply_activation(self, a):
        """
        Apply the specified activation function to the input.
        
        Args:
            a (np.ndarray): Input array to apply the activation function to.
            
        Returns:
            np.ndarray: Activated output.
        """
        if self.activation_function == 'sigmoid':
            return self.sigmoid(a)
        elif self.activation_function == 'relu':
            return self.relu(a)
        elif self.activation_function == 'tanh':
            return self.tanh(a)
        elif self.activation_function == 'linear':
            return self.linear(a)

    def get_activation_derivative(self, h):
        """
        Compute the derivative of the activation function based on the output.
        
        Args:
            h (np.ndarray): Output of the activation function.
            
        Returns:
            np.ndarray: Derivative of the activation function.
        """
        if self.activation_function == 'sigmoid':
            return h * (1 - h)
        elif self.activation_function == 'relu':
            return (h > 0).astype(float)
        elif self.activation_function == 'tanh':
            return 1 - h ** 2
        elif self.activation_function == 'linear':
            return np.ones_like(h)

    def sigmoid(self, x):
        """Sigmoid activation function, vectorized."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        """ReLU activation function, vectorized."""
        return np.maximum(0, x)

    def tanh(self, x):
        """Tanh activation function, vectorized."""
        return np.tanh(x)

    def linear(self, x):
        """Linear activation function (identity), vectorized."""
        return x

    def softmax(self, x):
        """Softmax activation function for the output layer, vectorized."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_pass(self, data_x, optimizer='rms_prop'):
        
        """
        Perform the forward pass through the network.
        
        Args:
            data_x (np.ndarray): Input data.
            optimizer (str): Optimizer type to determine weight updates.
            
        Returns:
            list: List of neuron outputs at each layer.
        """
        
        neuron_outputs = [data_x]
        if optimizer.lower() != 'nag':
            
            # Hidden layers
            for i in range(self.num_layers - 2):
                a = np.dot(neuron_outputs[-1], self.weights[i]) + self.biases[i]
                h = self.apply_activation(a)
                neuron_outputs.append(h)
                
            # Output layer
            a = np.dot(neuron_outputs[-1], self.weights[-1]) + self.biases[-1]
            output = self.softmax(a)
            neuron_outputs.append(output)
            
        else:  # NAG optimizer
            
            # Hidden layers
            for i in range(self.num_layers - 2):
                a = (np.dot(neuron_outputs[-1], self.weights[i] - self.beta_1 * self.weights_velocity[i]) +
                     self.biases[i] - self.beta_1 * self.bias_velocity[i])
                h = self.apply_activation(a)
                neuron_outputs.append(h)
                
            # Output layer
            a = (np.dot(neuron_outputs[-1], self.weights[-1] - self.beta_1 * self.weights_velocity[-1]) +
                 self.biases[-1] - self.beta_1 * self.bias_velocity[-1])
            output = self.softmax(a)
            neuron_outputs.append(output)
        return neuron_outputs

    def backward_pass(self, x, y, neuron_outputs, learning_rate, t, optimizer, loss_function, weight_decay):
        """
        Perform the backward pass to update weights and biases.
        
        Args:
            x (np.ndarray): Input batch.
            y (np.ndarray): True labels.
            neuron_outputs (list): Outputs from the forward pass.
            learning_rate (float): Learning rate for updates.
            t (int): Current iteration for bias correction in optimizers.
            optimizer (str): Optimizer type.
            loss_function (str): Loss function type.
            weight_decay (float): Weight decay parameter.
        """
        batch_size = len(x)
        # Compute initial delta for output layer
        if loss_function.lower() == 'categorical_cross_entropy':
            delta = neuron_outputs[-1] - y
        else:  # Mean squared error
            delta = np.zeros((batch_size, self.layer_sizes[-1]))
            for i in range(batch_size):
                softmax_jacobian = (np.diag(neuron_outputs[-1][i]) -
                                   np.outer(neuron_outputs[-1][i], neuron_outputs[-1][i]))
                delta[i] = 2 * np.dot(neuron_outputs[-1][i] - y[i], softmax_jacobian)

        # Update weights and biases layer by layer
        if optimizer.lower() != 'nag':
            
            for i in range(self.num_layers - 2, -1, -1):
                
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                
                if i > 0:
                    delta = np.dot(delta, self.weights[i].T) * self.get_activation_derivative(neuron_outputs[i])

                if optimizer.lower() == 'sgd':
                    self.weights[i] -= learning_rate * dw
                    self.biases[i] -= learning_rate * db
                    
                elif optimizer.lower() == 'momentum sgd':
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
                    self.weights[i] -= learning_rate * self.weights_velocity[i]
                    self.biases[i] -= learning_rate * self.bias_velocity[i]
                    
                elif optimizer.lower() == 'rms_prop':
                    self.history_weights[i] = (self.beta_1 * self.history_weights[i] +
                                              (1 - self.beta_1) * (dw ** 2))
                    self.history_bias[i] = (self.beta_1 * self.history_bias[i] +
                                           (1 - self.beta_1) * (db ** 2))
                    self.weights[i] -= learning_rate * dw / np.sqrt(self.history_weights[i] + 1e-12)
                    self.biases[i] -= learning_rate * db / np.sqrt(self.history_bias[i] + 1e-12)
                    
                elif optimizer.lower() == 'adam':
                    self.weights_velocity[i] = (self.beta_1 * self.weights_velocity[i] +
                                               (1 - self.beta_1) * dw)
                    self.bias_velocity[i] = (self.beta_1 * self.bias_velocity[i] +
                                            (1 - self.beta_1) * db)
                    self.history_weights[i] = (self.beta_2 * self.history_weights[i] +
                                              (1 - self.beta_2) * (dw ** 2))
                    self.history_bias[i] = (self.beta_2 * self.history_bias[i] +
                                           (1 - self.beta_2) * (db ** 2))
                    w_vel_corr = self.weights_velocity[i] / (1 - self.beta_1 ** t)
                    b_vel_corr = self.bias_velocity[i] / (1 - self.beta_1 ** t)
                    w_hist_corr = self.history_weights[i] / (1 - self.beta_2 ** t)
                    b_hist_corr = self.history_bias[i] / (1 - self.beta_2 ** t)
                    self.weights[i] -= learning_rate * w_vel_corr / np.sqrt(w_hist_corr + 1e-12)
                    self.biases[i] -= learning_rate * b_vel_corr / np.sqrt(b_hist_corr + 1e-12)
                    
                elif optimizer.lower() == 'nadam':
                    self.weights_velocity[i] = (self.beta_1 * self.weights_velocity[i] +
                                               (1 - self.beta_1) * dw)
                    self.bias_velocity[i] = (self.beta_1 * self.bias_velocity[i] +
                                            (1 - self.beta_1) * db)
                    self.history_weights[i] = (self.beta_2 * self.history_weights[i] +
                                              (1 - self.beta_2) * (dw ** 2))
                    self.history_bias[i] = (self.beta_2 * self.history_bias[i] +
                                           (1 - self.beta_2) * (db ** 2))
                    w_momentum = (self.beta_1 * self.weights_velocity[i] + (1 - self.beta_1) * dw) / (1 - self.beta_1 ** t)
                    b_momentum = (self.beta_1 * self.bias_velocity[i] + (1 - self.beta_1) * db) / (1 - self.beta_1 ** t)
                    w_hist_corr = self.history_weights[i] / (1 - self.beta_2 ** t)
                    b_hist_corr = self.history_bias[i] / (1 - self.beta_2 ** t)
                    self.weights[i] -= learning_rate * w_momentum / np.sqrt(w_hist_corr + 1e-12)
                    self.biases[i] -= learning_rate * b_momentum / np.sqrt(b_hist_corr + 1e-12)
                    
        else:  # NAG optimizer
            for i in range(self.num_layers - 2, -1, -1):
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                if i > 0:
                    delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * self.get_activation_derivative(neuron_outputs[i])
                self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
                self.weights[i] -= learning_rate * self.weights_velocity[i]
                self.biases[i] -= learning_rate * self.bias_velocity[i]

    def train(self, x_train, y_train, x_val, y_val, epochs=50, learning_rate=0.001, batch_size=32,
              optimizer="nag", beta_1=0.900, beta_2=0.999, loss_function='categorical_cross_entropy',
              weight_decay=0):
        
        """
        Train the neural network.
        
        Args:
            x_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training labels.
            x_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation labels.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            batch_size (int): Size of each mini-batch.
            optimizer (str): Optimizer type.
            beta_1 (float): Momentum parameter.
            beta_2 (float): RMSProp/Adam parameter.
            loss_function (str): Loss function type.
            weight_decay (float): Weight decay parameter.
        """
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        for epoch in range(epochs):
            indices = np.random.permutation(x_train.shape[0])
            
            x_train_permuted = x_train[indices]
            y_train_permuted = y_train[indices]
            total_loss = 0
            batch_num = 0
            
            for i in range(0, x_train.shape[0], batch_size):
                
                batch_x = x_train_permuted[i:i + batch_size]
                batch_y = y_train_permuted[i:i + batch_size]
                neuron_outputs = self.forward_pass(batch_x, optimizer)
                l2_norm_params = sum(np.sum(w ** 2) for w in self.weights) + sum(np.sum(b ** 2) for b in self.biases)
                if loss_function.lower() == 'categorical_cross_entropy':
                    loss = (-np.mean(np.sum(batch_y * np.log(neuron_outputs[-1] + 1e-10), axis=1)) +
                            (weight_decay / 2) * l2_norm_params)
                else:
                    loss = (np.mean(np.sum((batch_y - neuron_outputs[-1]) ** 2, axis=1)) +
                            (weight_decay / 2) * l2_norm_params)
                total_loss += loss
                batch_num += 1
                self.backward_pass(batch_x, batch_y, neuron_outputs, learning_rate, batch_num, optimizer,
                                  loss_function, weight_decay)
                
            average_loss = total_loss / batch_num
            validation_predictions = self.predict(x_val)
            validation_accuracy = np.mean(validation_predictions == np.argmax(y_val, axis=1))
            wandb.log({'epoch': epoch, 'train_loss': average_loss, 'val_accuracy': validation_accuracy})
            print(f"epoch: {epoch}, train_loss:{average_loss:.4f}, val_accuracy: {validation_accuracy:.4f}")

    def predict(self, x):
        """
        Predict class labels for the input data.
        
        Args:
            x (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Predicted class indices.
        """
        neuron_outputs = self.forward_pass(x)
        return np.argmax(neuron_outputs[-1], axis=1)

if __name__ == "__main__":
    
    wandb.login()
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Prepare data
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    train_size = 50000
    
    train_x = train_images[indices[:train_size]].reshape(-1, 784) / 255
    train_y = np.eye(10)[train_labels[indices[:train_size]]]
    val_x = train_images[indices[train_size:]].reshape(-1, 784) / 255
    val_y = np.eye(10)[train_labels[indices[train_size:]]]
    
    test_images = test_images.reshape(-1, 784) / 255
    test_labels = np.eye(10)[test_labels]

    # Initialize wandb
    wandb.init(project='backprop_scratch', config={
        'Learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32,
        'layer_size': [784, 128, 64, 10],
        'optimizer': 'Adam',
        'beta_1': 0.900,
        'beta_2': 0.999,
        'loss_function': 'categorical_cross_entropy',
        'weight_decay': 0.0005,
        'activation_function': 'relu'
    })
    
    config = wandb.config
    print(wandb.config)

    # Create and train the model
    model = backprop_from_scratch(config.layer_size, config.activation_function)
    model.train(train_x, train_y, val_x, val_y, config.epochs, config.Learning_rate, config.batch_size,
                config.optimizer, config.beta_1, config.beta_2, config.loss_function, config.weight_decay)

    # Evaluate on test set
    test_predictions = model.predict(test_images)
    test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis=1))
    print(f"test_accuracy: {test_accuracy:.4f}")
    wandb.log({'test_accuracy': test_accuracy})
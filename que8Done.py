# Import necessary libraries
import numpy as np
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import wandb
import argparse

# Define the neural network class
class backprop_from_scratch:
    
    def __init__(self, layer_size, activation_function='sigmoid', epsilon=1e-8, weight_init='random'):
        
        """
        Initialize the neural network with layer sizes, activation function, epsilon, and weight initialization.

        Args:
            layer_size (list): List of integers specifying the size of each layer (input, hidden, output).
            activation_function (str): Activation function for hidden layers ('sigmoid', 'relu', 'tanh', 'linear').
            epsilon (float): Small value for numerical stability in computations.
            weight_init (str): Weight initialization method ('random', 'Xavier').
        """
        
        # Validate and map activation function
        valid_activations = ['sigmoid', 'relu', 'tanh', 'linear']
        activation_map = {'identity': 'linear', 'ReLU': 'relu'}
        activation_function = activation_map.get(activation_function, activation_function).lower()
        if activation_function not in valid_activations:
            raise ValueError(f"Activation function must be one of {valid_activations}, got {activation_function}")
        
        self.activation_function = activation_function
        self.epsilon = epsilon
        self.weight_init = weight_init.lower()

        # Initialize network architecture
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_size)
        self.layer_sizes = layer_size
        self.initialize_params()

        # Initialize optimizer histories and velocities
        self.history_weights = [np.zeros_like(w) for w in self.weights]
        self.history_bias = [np.zeros_like(b) for b in self.biases]
        
        self.weights_velocity = [np.zeros_like(w) for w in self.weights]
        self.bias_velocity = [np.zeros_like(b) for b in self.biases]
        
        self.beta_1 = 0.900
        self.beta_2 = 0.999

    def initialize_params(self):
        
        """Initialize weights based on the specified method and biases with zeros."""
        
        gain_dict = {'sigmoid': 1, 'tanh': 5/3, 'relu': np.sqrt(2), 'linear': 1}
        for i in range(self.num_layers - 1):
            
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            if self.weight_init == 'random':
                w = np.random.randn(fan_in, fan_out) * 0.01
                
            elif self.weight_init == 'xavier':
                gain = gain_dict.get(self.activation_function, 1)
                std = gain * np.sqrt(2 / (fan_in + fan_out))
                w = np.random.randn(fan_in, fan_out) * std
                
            b = np.zeros((1, fan_out))
            
            self.weights.append(w)
            self.biases.append(b)

    def apply_activation(self, a):
        
        """Apply the specified activation function."""
        
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(a, -500, 500)))
        
        elif self.activation_function == 'relu':
            return np.maximum(0, a)
        
        elif self.activation_function == 'tanh':
            return np.tanh(a)
        
        elif self.activation_function == 'linear':
            return a

    def get_activation_derivative(self, h):
        
        """Compute the derivative of the activation function."""
        
        if self.activation_function == 'sigmoid':
            return h * (1 - h)
        
        elif self.activation_function == 'relu':
            return (h > 0).astype(float)
        
        elif self.activation_function == 'tanh':
            return 1 - h ** 2
        
        elif self.activation_function == 'linear':
            return np.ones_like(h)

    def softmax(self, x):
        
        """Softmax activation for the output layer."""
        
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_pass(self, data_x, optimizer='rms_prop'):
        
        """Perform the forward pass through the network."""
        
        neuron_outputs = [data_x]
        if optimizer.lower() != 'nag':
            
            for i in range(self.num_layers - 2):
                a = np.dot(neuron_outputs[-1], self.weights[i]) + self.biases[i]
                h = self.apply_activation(a)
                neuron_outputs.append(h)
                
            a = np.dot(neuron_outputs[-1], self.weights[-1]) + self.biases[-1]
            output = self.softmax(a)
            neuron_outputs.append(output)
            
        else:
            
            for i in range(self.num_layers - 2):
                
                a = (np.dot(neuron_outputs[-1], self.weights[i] - self.beta_1 * self.weights_velocity[i]) +
                     self.biases[i] - self.beta_1 * self.bias_velocity[i])
                
                h = self.apply_activation(a)
                neuron_outputs.append(h)
                
            a = (np.dot(neuron_outputs[-1], self.weights[-1] - self.beta_1 * self.weights_velocity[-1]) +
                 self.biases[-1] - self.beta_1 * self.bias_velocity[-1])
            
            output = self.softmax(a)
            neuron_outputs.append(output)
            
        return neuron_outputs

    def backward_pass(self, x, y, neuron_outputs, learning_rate, t, optimizer, loss_function, weight_decay):
        
        """Perform the backward pass to update weights and biases."""
        
        batch_size = len(x)
        if loss_function.lower() == 'categorical_cross_entropy':
            delta = neuron_outputs[-1] - y
            
        else:
            delta = np.zeros((batch_size, self.layer_sizes[-1]))
            for i in range(batch_size):
                softmax_jacobian = (np.diag(neuron_outputs[-1][i]) -
                                   np.outer(neuron_outputs[-1][i], neuron_outputs[-1][i]))
                delta[i] = 2 * np.dot(neuron_outputs[-1][i] - y[i], softmax_jacobian)

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
                    self.weights[i] -= learning_rate * dw / np.sqrt(self.history_weights[i] + self.epsilon)
                    self.biases[i] -= learning_rate * db / np.sqrt(self.history_bias[i] + self.epsilon)
                    
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
                    
                    self.weights[i] -= learning_rate * w_vel_corr / np.sqrt(w_hist_corr + self.epsilon)
                    self.biases[i] -= learning_rate * b_vel_corr / np.sqrt(b_hist_corr + self.epsilon)
                    
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
                    
                    self.weights[i] -= learning_rate * w_momentum / np.sqrt(w_hist_corr + self.epsilon)
                    self.biases[i] -= learning_rate * b_momentum / np.sqrt(b_hist_corr + self.epsilon)
        else:
            for i in range(self.num_layers - 2, -1, -1):
                
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                
                if i > 0:
                    delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * \
                            self.get_activation_derivative(neuron_outputs[i])
                            
                self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
                self.weights[i] -= learning_rate * self.weights_velocity[i]
                self.biases[i] -= learning_rate * self.bias_velocity[i]

    def train(self, x_train, y_train, x_val, y_val, epochs=50, learning_rate=0.001, batch_size=32,
              optimizer="nag", beta_1=0.900, beta_2=0.999, loss_function='categorical_cross_entropy',
              weight_decay=0):
        
        """Train the neural network with mini-batch gradient descent."""
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
                    
                    loss = (-np.mean(np.sum(batch_y * np.log(neuron_outputs[-1] + self.epsilon), axis=1)) +
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
            print(f"epoch: {epoch}, train_loss: {average_loss:.4f}, val_accuracy: {validation_accuracy:.4f}")

    def predict(self, x):
        
        """Predict class labels for input data."""
        neuron_outputs = self.forward_pass(x)
        return np.argmax(neuron_outputs[-1], axis=1)

# Main execution block
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a neural network with backpropagation from scratch.')
    
    parser.add_argument('-wp', '--wandb_project', type=str, default="myprojectname", help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train neural network')
    
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size used to train neural network')
    
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    
    parser.add_argument('-o', '--optimizer', type=str, default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer type')
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
    
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers')
    
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
    
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimizers')
    
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimizers')
    
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers')
    
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers')
    
    parser.add_argument('-w_i', '--weight_init', type=str, default='random', choices=['random', 'Xavier'], help='Weight initialization method')
    
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers used in feedforward neural network')
    
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons in a feedforward layer')
    
    parser.add_argument('-a', '--activation', type=str, default='sigmoid', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function')

    args = parser.parse_args()
    
    config = vars(args)
    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)

    def train():
        
        config = wandb.config
        
        if config.dataset == 'mnist':
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            
        elif config.dataset == 'fashion_mnist':
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

        train_images = train_images.reshape(-1, 784) / 255.0
        test_images = test_images.reshape(-1, 784) / 255.0
        train_labels = np.eye(10)[train_labels]
        test_labels = np.eye(10)[test_labels]

        indices = np.arange(train_images.shape[0])
        np.random.shuffle(indices)
        train_size = 50000
        train_x = train_images[indices[:train_size]]
        train_y = train_labels[indices[:train_size]]
        val_x = train_images[indices[train_size:]]
        val_y = train_labels[indices[train_size:]]

        # Map activation function
        if config.activation == 'identity':
            activation_function = 'linear'
        elif config.activation == 'ReLU':
            activation_function = 'relu'
        else:
            activation_function = config.activation

        # Map loss function
        if config.loss == 'cross_entropy':
            loss_function = 'categorical_cross_entropy'
        else:
            loss_function = config.loss

        # Map optimizer
        if config.optimizer == 'momentum':
            optimizer = 'momentum sgd'
        elif config.optimizer == 'rmsprop':
            optimizer = 'rms_prop'
        else:
            optimizer = config.optimizer

        # Set beta_1 and beta_2 based on optimizer
        if optimizer in ['momentum sgd', 'nag']:
            beta_1 = config.momentum
            beta_2 = 0.999  # Default, not used
            
        elif optimizer == 'rms_prop':
            beta_1 = config.beta
            beta_2 = 0.999  # Default, not used
            
        elif optimizer in ['adam', 'nadam']:
            beta_1 = config.beta1
            beta_2 = config.beta2
            
        else:  # sgd
            beta_1 = 0.0  # Not used
            beta_2 = 0.999  # Not used

        # Construct layer_size
        if config.num_layers > 0:
            hidden_layers = [config.hidden_size] * config.num_layers
        else:
            hidden_layers = []
            
        layer_size = [784] + hidden_layers + [10]

        model = backprop_from_scratch(layer_size, activation_function, config.epsilon, config.weight_init)
        model.train(train_x, train_y, val_x, val_y, config.epochs, config.learning_rate, config.batch_size,
                    optimizer, beta_1, beta_2, loss_function, config.weight_decay)

        test_predictions = model.predict(test_images)
        test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis=1))
        wandb.log({'test_accuracy': test_accuracy})
        print(f"test_accuracy: {test_accuracy:.4f}")

    train()
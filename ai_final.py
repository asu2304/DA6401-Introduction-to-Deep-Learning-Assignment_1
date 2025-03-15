# Import necessary libraries
import numpy as np
from keras.datasets import mnist  # Added to support MNIST dataset
from keras.datasets import fashion_mnist  # Original dataset import

import wandb  # For logging and sweep functionality
import argparse  # For parsing command-line arguments

# Define the neural network class
class backprop_from_scratch:
    
    def __init__(self, layer_size, activation_function='sigmoid', epsilon=1e-8):
        
        """
        Initialize the neural network with layer sizes, activation function, and epsilon.

        Args:
            layer_size (list): List of integers specifying the size of each layer (input, hidden, output).
            activation_function (str): Activation function for hidden layers ('sigmoid', 'relu', 'tanh', 'linear').
            epsilon (float): Small value for numerical stability in computations (e.g., avoiding log(0)).
        """
        
        # Validate activation function to ensure it’s supported
        valid_activations = ['sigmoid', 'relu', 'tanh', 'linear']
        if activation_function.lower() not in valid_activations:
            raise ValueError(f"Activation function must be one of {valid_activations}, got {activation_function}")
        self.activation_function = activation_function.lower()  # Store as lowercase for consistency
        self.epsilon = epsilon  # Store epsilon for use in loss and optimizer calculations
        
        # Initialize network architecture and parameters
        self.weights = []  # List to hold weight matrices
        self.biases = []  # List to hold bias vectors
        self.num_layers = len(layer_size)  # Total number of layers (input + hidden + output)
        self.layer_sizes = layer_size  # Store layer sizes for reference
        self.initialize_params()  # Initialize weights and biases
        
        # Initialize optimizer histories and velocities with zeros, matching weight/bias shapes
        self.history_weights = [np.zeros_like(w) for w in self.weights]  # For RMSProp/Adam history
        self.history_bias = [np.zeros_like(b) for b in self.biases]  # For RMSProp/Adam history
        self.weights_velocity = [np.zeros_like(w) for w in self.weights]  # For momentum/NAG
        self.bias_velocity = [np.zeros_like(b) for b in self.biases]  # For momentum/NAG
        
        # Default beta values for optimizers (will be overridden if provided)
        self.beta_1 = 0.900  # Momentum/RMSProp/Adam beta1
        self.beta_2 = 0.999  # Adam/Nadam beta2

    def initialize_params(self):
        """Initialize weights with He initialization and biases with zeros."""
        # Loop through layers to create weight matrices and bias vectors
        for i in range(self.num_layers - 1):
            # He initialization for weights: scales by sqrt(2/input_size) for ReLU compatibility
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i + 1]))  # Biases initialized to zero
            self.weights.append(w)  # Add weight matrix to list
            self.biases.append(b)  # Add bias vector to list

    def apply_activation(self, a):
        """
        Apply the specified activation function to the input.

        Args:
            a (np.ndarray): Input array (pre-activation values).

        Returns:
            np.ndarray: Activated output.
        """
        # Select activation function based on self.activation_function
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
        Compute the derivative of the activation function based on its output.

        Args:
            h (np.ndarray): Output of the activation function.

        Returns:
            np.ndarray: Derivative of the activation function.
        """
        # Compute derivative based on activation function type
        if self.activation_function == 'sigmoid':
            return h * (1 - h)  # Sigmoid derivative: h * (1 - h)
        elif self.activation_function == 'relu':
            return (h > 0).astype(float)  # ReLU derivative: 1 if h > 0, else 0
        elif self.activation_function == 'tanh':
            return 1 - h ** 2  # Tanh derivative: 1 - h^2
        elif self.activation_function == 'linear':
            return np.ones_like(h)  # Linear derivative: constant 1

    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow."""
        # Clip input to [-500, 500] to avoid exp overflow, then compute sigmoid
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)  # Return max(0, x) element-wise

    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)  # NumPy’s built-in tanh function

    def linear(self, x):
        """Linear (identity) activation function."""
        return x  # Simply return the input unchanged

    def softmax(self, x):
        """Softmax activation function for the output layer."""
        # Subtract max per row for numerical stability, then compute softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Normalize to sum to 1

    def forward_pass(self, data_x, optimizer='rms_prop'):
        """
        Perform the forward pass through the network.

        Args:
            data_x (np.ndarray): Input data (batch_size x input_size).
            optimizer (str): Optimizer type, affects NAG computation.

        Returns:
            list: Neuron outputs at each layer (including input and output).
        """
        neuron_outputs = [data_x]  # Start with input layer
        if optimizer.lower() != 'nag':
            # Standard forward pass for non-NAG optimizers
            # Hidden layers
            for i in range(self.num_layers - 2):
                a = np.dot(neuron_outputs[-1], self.weights[i]) + self.biases[i]  # Linear transformation
                h = self.apply_activation(a)  # Apply activation
                neuron_outputs.append(h)  # Store output
            # Output layer
            a = np.dot(neuron_outputs[-1], self.weights[-1]) + self.biases[-1]  # Linear transformation
            output = self.softmax(a)  # Apply softmax for probabilities
            neuron_outputs.append(output)  # Store output
        else:
            # NAG forward pass: look-ahead using velocity
            # Hidden layers
            for i in range(self.num_layers - 2):
                # Adjust weights and biases with momentum terms
                a = (np.dot(neuron_outputs[-1], self.weights[i] - self.beta_1 * self.weights_velocity[i]) +
                     self.biases[i] - self.beta_1 * self.bias_velocity[i])
                h = self.apply_activation(a)  # Apply activation
                neuron_outputs.append(h)  # Store output
            # Output layer
            a = (np.dot(neuron_outputs[-1], self.weights[-1] - self.beta_1 * self.weights_velocity[-1]) +
                 self.biases[-1] - self.beta_1 * self.bias_velocity[-1])
            output = self.softmax(a)  # Apply softmax
            neuron_outputs.append(output)  # Store output
        return neuron_outputs

    def backward_pass(self, x, y, neuron_outputs, learning_rate, t, optimizer, loss_function, weight_decay):
        """
        Perform the backward pass to update weights and biases.

        Args:
            x (np.ndarray): Input batch.
            y (np.ndarray): True labels (one-hot encoded).
            neuron_outputs (list): Outputs from forward pass.
            learning_rate (float): Learning rate for updates.
            t (int): Current iteration (for bias correction in Adam/Nadam).
            optimizer (str): Optimizer type (sgd, momentum sgd, nag, rms_prop, adam, nadam).
            loss_function (str): Loss function (categorical_cross_entropy or mean_squared_error).
            weight_decay (float): L2 regularization parameter.
        """
        batch_size = len(x)  # Number of samples in the batch
        # Compute initial delta for output layer
        if loss_function.lower() == 'categorical_cross_entropy':
            delta = neuron_outputs[-1] - y  # Gradient of cross-entropy w.r.t. softmax output
        else:  # Mean squared error
            delta = np.zeros((batch_size, self.layer_sizes[-1]))  # Initialize delta array
            for i in range(batch_size):
                # Compute softmax Jacobian for MSE gradient
                softmax_jacobian = (np.diag(neuron_outputs[-1][i]) -
                                   np.outer(neuron_outputs[-1][i], neuron_outputs[-1][i]))
                delta[i] = 2 * np.dot(neuron_outputs[-1][i] - y[i], softmax_jacobian)  # MSE gradient

        # Update weights and biases from output to input layer
        if optimizer.lower() != 'nag':
            for i in range(self.num_layers - 2, -1, -1):  # Iterate backwards
                # Compute gradients with weight decay
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                if i > 0:  # If not the input layer
                    # Propagate delta backwards
                    delta = np.dot(delta, self.weights[i].T) * self.get_activation_derivative(neuron_outputs[i])

                # Update parameters based on optimizer
                if optimizer.lower() == 'sgd':
                    self.weights[i] -= learning_rate * dw  # Simple gradient descent
                    self.biases[i] -= learning_rate * db
                elif optimizer.lower() == 'momentum sgd':
                    # Momentum updates
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
                    self.weights[i] -= learning_rate * self.weights_velocity[i]
                    self.biases[i] -= learning_rate * self.bias_velocity[i]
                elif optimizer.lower() == 'rms_prop':
                    # RMSProp: update history with moving average of squared gradients
                    self.history_weights[i] = (self.beta_1 * self.history_weights[i] +
                                              (1 - self.beta_1) * (dw ** 2))
                    self.history_bias[i] = (self.beta_1 * self.history_bias[i] +
                                           (1 - self.beta_1) * (db ** 2))
                    # Update weights with stabilized denominator
                    self.weights[i] -= learning_rate * dw / np.sqrt(self.history_weights[i] + self.epsilon)
                    self.biases[i] -= learning_rate * db / np.sqrt(self.history_bias[i] + self.epsilon)
                elif optimizer.lower() == 'adam':
                    # Adam: update velocity (first moment) and history (second moment)
                    self.weights_velocity[i] = (self.beta_1 * self.weights_velocity[i] +
                                               (1 - self.beta_1) * dw)
                    self.bias_velocity[i] = (self.beta_1 * self.bias_velocity[i] +
                                            (1 - self.beta_1) * db)
                    self.history_weights[i] = (self.beta_2 * self.history_weights[i] +
                                              (1 - self.beta_2) * (dw ** 2))
                    self.history_bias[i] = (self.beta_2 * self.history_bias[i] +
                                           (1 - self.beta_2) * (db ** 2))
                    # Bias-corrected estimates
                    w_vel_corr = self.weights_velocity[i] / (1 - self.beta_1 ** t)
                    b_vel_corr = self.bias_velocity[i] / (1 - self.beta_1 ** t)
                    w_hist_corr = self.history_weights[i] / (1 - self.beta_2 ** t)
                    b_hist_corr = self.history_bias[i] / (1 - self.beta_2 ** t)
                    # Update weights with stabilized denominator
                    self.weights[i] -= learning_rate * w_vel_corr / np.sqrt(w_hist_corr + self.epsilon)
                    self.biases[i] -= learning_rate * b_vel_corr / np.sqrt(b_hist_corr + self.epsilon)
                elif optimizer.lower() == 'nadam':
                    # Nadam: Nesterov + Adam
                    self.weights_velocity[i] = (self.beta_1 * self.weights_velocity[i] +
                                               (1 - self.beta_1) * dw)
                    self.bias_velocity[i] = (self.beta_1 * self.bias_velocity[i] +
                                            (1 - self.beta_1) * db)
                    self.history_weights[i] = (self.beta_2 * self.history_weights[i] +
                                              (1 - self.beta_2) * (dw ** 2))
                    self.history_bias[i] = (self.beta_2 * self.history_bias[i] +
                                           (1 - self.beta_2) * (db ** 2))
                    # Nesterov momentum with bias correction
                    w_momentum = (self.beta_1 * self.weights_velocity[i] + (1 - self.beta_1) * dw) / (1 - self.beta_1 ** t)
                    b_momentum = (self.beta_1 * self.bias_velocity[i] + (1 - self.beta_1) * db) / (1 - self.beta_1 ** t)
                    w_hist_corr = self.history_weights[i] / (1 - self.beta_2 ** t)
                    b_hist_corr = self.history_bias[i] / (1 - self.beta_2 ** t)
                    # Update weights with stabilized denominator
                    self.weights[i] -= learning_rate * w_momentum / np.sqrt(w_hist_corr + self.epsilon)
                    self.biases[i] -= learning_rate * b_momentum / np.sqrt(b_hist_corr + self.epsilon)
        else:  # NAG optimizer
            for i in range(self.num_layers - 2, -1, -1):
                # Compute gradients with weight decay
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                if i > 0:
                    # Propagate delta with NAG-adjusted weights
                    delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * \
                            self.get_activation_derivative(neuron_outputs[i])
                # Update velocities and parameters
                self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
                self.weights[i] -= learning_rate * self.weights_velocity[i]
                self.biases[i] -= learning_rate * self.bias_velocity[i]

    def train(self, x_train, y_train, x_val, y_val, epochs=50, learning_rate=0.001, batch_size=32,
              optimizer="nag", beta_1=0.900, beta_2=0.999, loss_function='categorical_cross_entropy',
              weight_decay=0):
        """
        Train the neural network with mini-batch gradient descent.

        Args:
            x_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training labels (one-hot encoded).
            x_val (np.ndarray): Validation input data.
            y_val (np.ndarray): Validation labels (one-hot encoded).
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for updates.
            batch_size (int): Size of each mini-batch.
            optimizer (str): Optimizer type.
            beta_1 (float): Momentum/RMSProp/Adam parameter.
            beta_2 (float): Adam/Nadam parameter.
            loss_function (str): Loss function type.
            weight_decay (float): L2 regularization parameter.
        """
        # Set optimizer parameters
        self.beta_1 = beta_1  # Update instance beta_1
        self.beta_2 = beta_2  # Update instance beta_2
        for epoch in range(epochs):
            # Shuffle training data for each epoch
            indices = np.random.permutation(x_train.shape[0])
            x_train_permuted = x_train[indices]
            y_train_permuted = y_train[indices]
            total_loss = 0  # Accumulate loss over batches
            batch_num = 0  # Count batches for averaging
            # Mini-batch training
            for i in range(0, x_train.shape[0], batch_size):
                batch_x = x_train_permuted[i:i + batch_size]  # Extract batch input
                batch_y = y_train_permuted[i:i + batch_size]  # Extract batch labels
                neuron_outputs = self.forward_pass(batch_x, optimizer)  # Forward pass
                # Compute L2 regularization term
                l2_norm_params = sum(np.sum(w ** 2) for w in self.weights) + sum(np.sum(b ** 2) for b in self.biases)
                # Compute loss with epsilon for stability
                if loss_function.lower() == 'categorical_cross_entropy':
                    loss = (-np.mean(np.sum(batch_y * np.log(neuron_outputs[-1] + self.epsilon), axis=1)) +
                            (weight_decay / 2) * l2_norm_params)
                else:  # Mean squared error
                    loss = (np.mean(np.sum((batch_y - neuron_outputs[-1]) ** 2, axis=1)) +
                            (weight_decay / 2) * l2_norm_params)
                total_loss += loss  # Add to total loss
                batch_num += 1  # Increment batch counter
                # Backward pass to update weights
                self.backward_pass(batch_x, batch_y, neuron_outputs, learning_rate, batch_num, optimizer,
                                  loss_function, weight_decay)
            # Compute average loss for the epoch
            average_loss = total_loss / batch_num
            # Evaluate on validation set
            validation_predictions = self.predict(x_val)
            validation_accuracy = np.mean(validation_predictions == np.argmax(y_val, axis=1))
            # Log metrics to wandb
            wandb.log({'epoch': epoch, 'train_loss': average_loss, 'val_accuracy': validation_accuracy})
            # Print progress
            print(f"epoch: {epoch}, train_loss: {average_loss:.4f}, val_accuracy: {validation_accuracy:.4f}")

    def predict(self, x):
        """
        Predict class labels for input data.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class indices.
        """
        neuron_outputs = self.forward_pass(x)  # Forward pass
        return np.argmax(neuron_outputs[-1], axis=1)  # Return class with highest probability


# Main execution block
if __name__ == "__main__":
    # Set up command-line argument parser
    
    parser = argparse.ArgumentParser(description='Train a neural network with backpropagation from scratch.')
    
    # setting the project name 
    parser.add_argument('-wp', '--wandb_project', type='str', default="back_from_full_scratch", help='give project name for wandb')
    
    # setting up wandb entity name
    parser.add_argument('-we', '--wandb_entity', type='str', default='myname', help='wandb for tracking experiments')
    
    # Dataset choice: mnist or fashion_mnist
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use (mnist or fashion_mnist)')
    
    # Number of epochs
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    # Batch size
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Size of each mini-batch')

    # Loss function choice
    parser.add_argument('-l', '--loss_function', type=str, default='categorical_cross_entropy',
                        choices=['categorical_cross_entropy', 'mean_squared_error'],
                        help='Loss function type')

    # Optimizer choice
    parser.add_argument('-o', '--optimizer', type=str, default='rms_prop',
                        choices=['sgd', 'momentum sgd', 'nag', 'rms_prop', 'adam', 'nadam'],
                        help='Optimizer type')

    # Learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimization')


    # taking beta_1 for momentum and nestrov, later we set that equal to beta_1 as per our implemenation
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum for momentum-based optimizers')

    # taking beta for rms prop and we will set it equal to beta_1 becuase we have used beta_1 in our implementation
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta for rmsprop optimizer')
    
    # taking beta_1 for 
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 for adam and nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 for adam and nadam')

    # Hidden layer sizes as a list (e.g., --hidden_sizes 128 64 for two layers)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64],
                        help='List of hidden layer sizes (e.g., 128 64)')
    
    # Activation function choice
    parser.add_argument('--activation_function', type=str, default='relu',
                        choices=['sigmoid', 'relu', 'tanh', 'linear'],
                        help='Activation function for hidden layers')

    
    # Beta1 for optimizers
    parser.add_argument('--beta_1', type=float, default=0.900,
                        help='Beta1 parameter for momentum/RMSProp/Adam')
    # Beta2 for Adam/Nadam
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='Beta2 parameter for Adam/Nadam')
    
    # Weight decay (L2 regularization)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization) parameter')
    # Epsilon for numerical stability
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon value for numerical stability')

    # Parse arguments from command line
    args = parser.parse_args()

    # Convert parsed arguments to a dictionary for wandb config
    
    config = vars(args)
    # Initialize wandb with the config; for sweeps, wandb will override specified hyperparameters
    wandb.init(project=config['wandb_project'], config=config['wandb_entity'])

    def train():
        
        """Train the model using hyperparameters from wandb.config."""
        
        config = wandb.config  # Access hyperparameters (from command line or sweep)
        # Load dataset based on config.dataset
        
        if config.dataset == 'mnist':
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            
        elif config.dataset == 'fashion_mnist':
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
            
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

        # Preprocess data: flatten and normalize images, one-hot encode labels
        train_images = train_images.reshape(-1, 784) / 255.0  # Flatten to 784, normalize to [0, 1]
        test_images = test_images.reshape(-1, 784) / 255.0  # Same for test set
        train_labels = np.eye(10)[train_labels]  # One-hot encode training labels
        test_labels = np.eye(10)[test_labels]  # One-hot encode test labels

        # Split training data into train and validation sets (50000 train, 10000 val)
        indices = np.arange(train_images.shape[0])
        np.random.shuffle(indices)  # Randomize indices
        train_size = 50000  # Fixed split as in original code
        train_x = train_images[indices[:train_size]]  # Training input
        train_y = train_labels[indices[:train_size]]  # Training labels
        val_x = train_images[indices[train_size:]]  # Validation input
        val_y = train_labels[indices[train_size:]]  # Validation labels

        # Construct layer_size: input (784), hidden layers, output (10)
        layer_size = [784] + config.hidden_sizes + [10]  # Flexible hidden layers from config
        
        # Create model instance with specified parameters
        model = backprop_from_scratch(layer_size, config.activation_function, config.epsilon)
        
        config.beta_1 = config.momentum # setting config.beta_1 as same as that of momentum because in our implementation we have implemented momentum and nestrov with beta_1 only
        config.beta_1 = config.beta
        # Train the model with all hyperparameters from config
        model.train(train_x, train_y, val_x, val_y, config.epochs, config.learning_rate, config.batch_size,
                    config.optimizer, config.beta_1, config.beta_2, config.loss_function, config.weight_decay)

        # Evaluate on test set
        test_predictions = model.predict(test_images)  # Get predictions
        test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis=1))  # Compute accuracy
        wandb.log({'test_accuracy': test_accuracy})  # Log test accuracy to wandb
        print(f"test_accuracy: {test_accuracy:.4f}")  # Print test accuracy

    # Execute training
    train()
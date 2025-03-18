
import numpy as np # for matrix multiplications
from keras.datasets import mnist, fashion_mnist # used just to get the data
import wandb
import argparse # to take command line arguement
import matplotlib.pyplot as plt # for consusion matrix

# defining a backprop_from_scratch to do all.
class backprop_from_scratch:

    # define constructor
    def __init__(self, layer_size, activation_function = 'ReLu', initialization = 'xavier'):
        
        # initialise the neural network
        self.weights, self.biases = [], []
        self.num_layers = len(layer_size)
        self.layer_sizes = layer_size
        self.activation_function = activation_function
        self.initialization = initialization
        self.initialize_params()
        
         # declaring and initializing the history for weights and bias
        self.history_weights = []
        self.history_bias = []

        self.history_weights = [np.zeros_like(w) for w in self.weights]
        self.history_bias = [np.zeros_like(b) for b in self.biases]

        # delarin and initializing the velocity for weights and bias
        self.weights_velocity = []
        self.bias_velocity = []

        self.weights_velocity = [np.zeros_like(w) for w in self.weights]
        self.bias_velocity = [np.zeros_like(b) for b in self.biases]

        # setting the beta with there default values
        self.beta_1 = 0.900
        self.beta_2 = 0.999
        
    def initialize_params(self):
        # let's use He initialization for weights and set values of bias equals to zero
        for i in range(self.num_layers-1):
            if self.initialization == 'xavier':
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(1/self.layer_sizes[i])
            elif self.initialization == 'random': 
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) 
            else: 
                raise ValueError('invalid initialization method passed')
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w) 
            self.biases.append(b) 
            
        # print(self.weights)
        # print(self.biases)

    def sigmoid(self, X):
        # clipping it bw -500 to 500 
        return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x*(1-x) 

    def relu(self, x):
        return np.maximum(0,x)

    def relu_derivative(self, x): 
        return np.where(x>0, 1, 0)

    def tanh(self, x): 
       return np.tanh(x) 
        
    def tanh_derivative(self, x): 
        return 1 - x ** 2
    
    def softmax(self, X): 
        # clipping it for numerical stability
        exp_x = np.exp(X - np.max(X, axis = 1, keepdims=True))
        return exp_x/np.sum(exp_x, axis = 1, keepdims=True)
        
    def forward_pass(self, data_X, Learning_rate, optimizer='RMS_Prop'): # setting defalut optimizer as RMS_Prop cauz while infering we 
        # used this function meaning we dont want updated forward pass for NAG, while inferring.
        
        neuron_outputs = [data_X]
        # pass thorugh hidden layers
        if optimizer != 'NAG':
            for i in range(self.num_layers - 2): 
        
                
                a = np.dot(neuron_outputs[-1], self.weights[i]) + self.biases[i]
                if self.activation_function == 'ReLu':
                    h = self.relu(a)
                elif self.activation_function == 'Sigmoid':
                    h = self.sigmoid(a)
                elif self.activation_function == 'tanh':
                    h = self.tanh(a)
                elif self.activation_function == 'Linear':
                    h = a
                else: 
                    raise ValueError('invalide activation passed')
                    
                neuron_outputs.append(h)
                
            # pass thorugh the output layer
            a = np.dot(neuron_outputs[-1], self.weights[-1]) + self.biases[-1]
            output = self.softmax(a)
            neuron_outputs.append(output)
            return neuron_outputs
        
        else: # if optimizer is NAG
            for i in range(self.num_layers - 2): 

                
                a = np.dot(neuron_outputs[-1], self.weights[i] - self.beta_1 * self.weights_velocity[i] * Learning_rate)  # change
                + self.biases[i] - self.beta_1 * self.bias_velocity[i] 
                
                if self.activation_function == 'ReLu':
                    h = self.relu(a)
                elif self.activation_function == 'Sigmoid':
                    h = self.sigmoid(a)
                elif self.activation_function == 'tanh':
                    h = self.tanh(a)
                elif self.activation_function == 'Linear':
                    h = a
                else: 
                    raise ValueError('invalide activation passed')     
                    
                neuron_outputs.append(h)
                
            # pass thorugh the output layer
            a = np.dot(neuron_outputs[-1], self.weights[-1] - self.beta_1 * self.weights_velocity[-1] * Learning_rate) # change
            + self.biases[-1] - self.beta_1 * self.bias_velocity[-1]
            
            output = self.softmax(a)
            neuron_outputs.append(output)
            return neuron_outputs

    
    def backword_pass(self, X, y, neuron_outputs, learning_rate, t, optimizer, loss_function, weight_decay, epsilon): 
        
        # compute the gradient at given value of params
        batch_size = len(X)
        
        # computing gradies with repect to the output layer
        if loss_function == 'categorical_cross_entropy':
            delta = neuron_outputs[-1] - y # shape of delta is 32 x 10
            
        else:
            
            batch_size = len(neuron_outputs[-1])
            classes = len(neuron_outputs[-1][0])
            # size mismatch problem here
            delta = np.zeros((batch_size, classes))
            
            for i in range(batch_size):
                softmax_jacobian = np.diag(neuron_outputs[-1][i]) - np.outer(neuron_outputs[-1][i], neuron_outputs[-1][i])
                # print(softmax_jacobian.shape)
                # print(neuron_output[-1][i]
                delta[i] = 2 * np.dot(neuron_outputs[-1][i] - y[i], softmax_jacobian)
                
        if optimizer != 'NAG':
            
            # computing gradient with respect to hidden layers
            for i in range(self.num_layers - 2, -1, -1):
                
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                
                if i>0: # because computing delta for i=0 will be absolutely un-necessary.
                    if self.activation_function == 'ReLu': 
                        delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(neuron_outputs[i])
                    elif self.activation_function == 'Sigmoid': 
                        delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(neuron_outputs[i])
                    elif self.activation_function == 'tanh': 
                        delta = np.dot(delta, self.weights[i].T) * self.tanh_derivative(neuron_outputs[i])
                    elif self.activation_function == 'Linear': 
                        delta = np.dot(delta, self.weights[i].T) * np.ones_like(neuron_outputs[i])
    
                if optimizer == 'sgd':
    
                    # making an update
                    self.weights[i] -= dw * learning_rate
                    self.biases[i] -= db * learning_rate
    
                    
                elif optimizer == 'momentum sgd':
                    
                    # computing the momentum
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
            
                    # making an update
                    self.weights[i] -= self.weights_velocity[i] * learning_rate
                    self.biases[i] -= self.bias_velocity[i] * learning_rate
    
                elif optimizer == 'RMS_Prop':
                    
                    # we need to change the learning rate at each iteration as per history accumulated for the current layer w and b
                    self.history_weights[i] = self.beta_1 * self.history_weights[i] + (1 - self.beta_1) * (dw ** 2)
                    self.history_bias[i] = self.beta_1 * self.history_bias[i] + (1 - self.beta_1) * (db ** 2) 
                    self.weights[i] -= dw * learning_rate / np.sqrt(self.history_weights[i] + epsilon)
                    self.biases[i] -= db * learning_rate / np.sqrt(self.history_bias[i] + epsilon)
    
                elif optimizer == 'Adam':
                    
                    # adding the momentum
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + (1 - self.beta_1) * dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + (1 - self.beta_1) * db
                    
                    # we need to change the learning rate at each iteration as per history accumulated for the current layer w and b
                    self.history_weights[i] = self.beta_2 * self.history_weights[i] + (1 - self.beta_2) * (dw ** 2)
                    self.history_bias[i] = self.beta_2 * self.history_bias[i] + (1 - self.beta_2) * (db ** 2) 
        
                    # implementing the adam rule for update with bias corrected
                    self.weights[i] -= self.weights_velocity[i]/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_weights[i] / (1 - self.beta_2** t) + epsilon)
                    self.biases[i] -= self.bias_velocity[i]/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_bias[i] / (1 - self.beta_2 ** t) + epsilon) 
    
                elif optimizer == 'Nadam':
                    
                    # adding the momentum
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + (1 - self.beta_1) * dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + (1 - self.beta_1) * db
        
                    # we need to change the learning rate at each iteration as per history accumulated for the current layer w&b
                    self.history_weights[i] = self.beta_2 * self.history_weights[i] + (1 - self.beta_2) * (dw ** 2)
                    self.history_bias[i] = self.beta_2 * self.history_bias[i] + (1 - self.beta_2) * (db ** 2) 
                    
                    # implementing the nadam rule for update with bias corrected
                    self.weights[i] -= ((self.beta_1 * self.weights_velocity[i]) + (1 - self.beta_1) * dw)/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_weights[i] / (1 - self.beta_2** t) + epsilon)
                    self.biases[i] -= ((self.beta_1 * self.bias_velocity[i]) + (1 - self.beta_1) * db)/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_bias[i] / (1 - self.beta_2 ** t) + epsilon) 
                    
        else: # if optimizer is 'NAG'
            
            for i in range(self.num_layers - 2, -1, -1):
                
                # neuron_ouputs depend on the data, which we have computed in the forward pass
                # but delta will depend on on the weight matrices
                
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
                
                if i>0: # because computing delta for i=0 will be absolutely un-necessary.
                    if self.activation_function == 'ReLu': # changed 4 places
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i] * learning_rate).T) * self.relu_derivative(neuron_outputs[i])
                    elif self.activation_function == 'Sigmoid':
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i] * learning_rate).T) * self.sigmoid_derivative(neuron_outputs[i])
                    elif self.activation_function == 'tanh':
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i] * learning_rate).T) * self.tanh_derivative(neuron_outputs[i])    
                    elif self.activation_function == 'Linear':
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i] * learning_rate).T) * np.ones_like(neuron_outputs[i])
                        
                # computing the momentum
                self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
    
                # making an update
                self.weights[i] -= self.weights_velocity[i] * learning_rate
                self.biases[i] -= self.bias_velocity[i] * learning_rate
    
    # function to train the network
    def train(self, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001, batch_size=32, optimizer="Adam", beta_1=0.900, beta_2=0.999, loss_function='categorical_cross_entropy', weight_decay = 0, activation_function='ReLu', epsilon = 1e-6): # setting RMS_Prop as default optimizer
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        for epoch in range(epochs):
            # first shuffle the training data
            # print(X_train.shape[0])
            
            indices = np.random.permutation(X_train.shape[0]) 
            X_train_permuted = X_train[indices]
            y_train_permuted = y_train[indices]
            
            total_loss = 0
            batch_num = 0
            
            # let's make a update for a mini batch
            for i in range(0, X_train.shape[0], batch_size): 
                batch_x = X_train_permuted[i: i + batch_size]
                batch_y = y_train_permuted[i: i + batch_size]

                # calling forward and remember it's different for NAG so passing optmizer in it, to check inside it it's NAG
                neuron_outputs = self.forward_pass(batch_x, learning_rate, optimizer)
                
                l2_norm_weights, l2_norm_bias = 0, 0
                for i in range(len(self.weights)):
                    l2_norm_weights += np.sum(self.weights[i] ** 2)
                for i in range(len(self.biases)):
                    l2_norm_bias += np.sum(self.biases[i] ** 2) 
                l2_norm_params = l2_norm_bias + l2_norm_weights
                
                if loss_function == 'categorical_cross_entropy':
                    loss = -np.mean(np.sum(batch_y * np.log(neuron_outputs[-1] + 1e-10), axis = 1)) +  (weight_decay/2) * l2_norm_params 
                    # added 1e-10 to prevent num underflow
                    # this loss in not interpretable in case of NAG as this will compute the forward pass for update weights
                else:
                    loss = np.mean(np.sum((batch_y - neuron_outputs[-1]) ** 2, axis = 1)) + (weight_decay/2) * l2_norm_params
                    # loss function for MSE
            
                total_loss += loss
                batch_num += 1
                
                # backprop: computing the gradient and making an update in backwork pass func
                self.backword_pass(batch_x, batch_y, neuron_outputs, learning_rate, batch_num, optimizer, loss_function, weight_decay, epsilon)
            
            average_loss = total_loss/batch_num
            
            # now let's make predictions on validation dataset 
            validation_predictions = self.predict(X_val, learning_rate)
            validation_accuracy = np.mean(validation_predictions == np.argmax(y_val, axis = 1))

            # loging to wandb
            wandb.log({'epoch': epoch , 'train_loss': average_loss, 'val_accuracy': validation_accuracy})
            print(f"epoch: {epoch}, train_loss:{average_loss:.4f}, val_accuracy: {validation_accuracy:.4f}")
            
    def predict(self, X,learning_rate): 
        # this will predict class labels for the passed data
        neuron_outputs = self.forward_pass(X, learning_rate)
        return np.argmax(neuron_outputs[-1], axis=1)

if __name__ == "__main__":

    wandb.login()
    
    parser = argparse.ArgumentParser(description='Train a neural network with backpropagation from scratch.')
    
    parser.add_argument('-wp', '--wandb_project', type=str, default="confusion_matrix_for_the_best_combo", help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-we', '--wandb_entity', type=str, default='da24s006-indian-institue-of-technology-madras-', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use')
    
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network')
    
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size used to train neural network')
    
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer type')
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate used to optimize model parameters')
    
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers')
    
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
    
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers')
    
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers')
    
    parser.add_argument('-eps', '--epsilon', type=float, default=0.0000000001, help='Epsilon used by optimizers')
    
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers')
    
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier', choices=['random', 'Xavier'], help='Weight initialization method')
    
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers used in feedforward neural network')
    
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer')
    
    parser.add_argument('-a', '--activation', type=str, default='ReLU', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function')

    args = parser.parse_args()
    config = vars(args)

    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)

    if config['dataset'] == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif config['dataset'] == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Splitting the trainig data into train and validation
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    train_size = int(0.9 * len(train_images)) # using
    
    train_x = train_images[indices[:train_size]]
    train_y = train_labels[indices[:train_size]]
    val_x = train_images[indices[train_size:]]
    val_y = train_labels[indices[train_size:]]

    train_x = train_x.reshape(train_x.shape[0], -1) / 255
    val_x = val_x.reshape(val_x.shape[0], -1) / 255

    # converting y's into one hot vector
    num_classes = 10
    train_y = np.eye(num_classes)[train_y]
    val_y = np.eye(num_classes)[val_y]

    # let's do it for test data as well
    test_images = test_images.reshape(test_images.shape[0], -1) / 255
    test_labels = np.eye(num_classes)[test_labels]

    # making the passed arguement names consistent with the code i wrote 

    # setting momentum and beta equal to beta1 as per my implementation
    if config['optimizer'] == 'momentum':
        config['beta1'] = config['momentum']
    elif config['optimizer'] == 'rmsprop':
        config['beta1'] = config['beta']

    # loss function
    if config['loss'] == 'mean_squared_error':
        config['loss'] = 'mean_squared'
    else: 
        config['loss'] = 'categorical_cross_entropy'

    # optimizers
    # ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
    # sgd name is same
    if config['optimizer'] == 'momentum':
        config['optimizer'] = 'momentum sgd'
    elif config['optimizer'] == 'nag':
        config['optimizer'] = 'NAG'
    elif config['optimizer'] == 'rmsprop':
        config['optimizer'] = 'RMS_Prop'
    elif config['optimizer'] == 'adam':
        config['optimizer'] = 'Adam'
    elif config['optimizer'] == 'nadam':
        config['optimizer'] = 'Nadam'

    # initialization
    if config['weight_init'] == 'Xavier':
        config['weight_init'] = 'xavier'
    # same name for random 

    # activation function
    # ['identity', 'sigmoid', 'tanh', 'ReLU']
    if config['activation'] == 'ReLU':
        config['activation'] = 'ReLu'
    elif config['activation'] == 'sigmoid':
        config['activation'] = 'Sigmoid'
    elif config['activation'] == 'identity':
        config['activation'] = 'Linear'
        # same name for tanh

    
    # now let's create and train the network
    model = backprop_from_scratch([784] + [config['hidden_size']] * config['num_layers'] + [10], 
                                  config['activation'], 
                                  config['weight_init'])
    
    model.train(train_x, 
                train_y, 
                val_x, 
                val_y, 
                config['epochs'], 
                config['learning_rate'], 
                config['batch_size'], 
                config['optimizer'], 
                config['beta1'], 
                config['beta2'], 
                config['loss'], 
                config['weight_decay'], 
                config['activation'],
                config['epsilon']
               )

    fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
    }
    
    # now let's evaluate on the test set
    test_predictions = model.predict(test_images, config['learning_rate']) 
    test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis = 1))
    print(f"test_accuracy:{test_accuracy: .4f}") 
    wandb.log({'test_accuracy': test_accuracy})

    true_labels = np.argmax(test_labels, axis=1)
    pred_labels = test_predictions
    num_classes = len(np.unique(true_labels))
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populating the confusion matrix
    for true, pred in zip(true_labels, pred_labels):
        conf_matrix[true, pred] += 1
        
    # Loging confusion matrix to WandB
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_labels,
        preds=pred_labels,
        class_names=[fashion_mnist_labels[i] for i in range(num_classes)]
    )})



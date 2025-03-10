import numpy as np
from keras.datasets import fashion_mnist # used just to get the data
import wandb

# defining a backprop_from_scratch to do all.
class backprop_from_scratch:

    # define constructor
    def __init__(self, layer_size, activation_function='ReLu', beta_1 = 0.900, beta_2 = 0.999, initialization = 'He'):
        
        # initialise the neural network
        self.weights, self.biases = [], []
        self.num_layers = len(layer_size)
        self.layer_sizes = layer_size
        self.activation_function = activation_function
        self.initialization = initialization
        self.initialize_params(initialization)
        
         # declarin and initializing the history for weights and bias
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def initialize_params(self, initialization):
        # let's use He initialization for weights and set values of bias equals to zero
        
        for i in range(self.num_layers-1):
            if initialization == 'He':
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2/self.layer_sizes[i])
            elif initialization == 'Xavier':
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(1/self.layer_sizes[i])    
            else:
                raise ValueError("invalid initialization method passed!")
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w) 
            self.biases.append(b) 
            
            
    def forward_pass(self, data_X, optimizer='RMS_Prop', activation_function='ReLu'): # setting defalut optimizer as RMS_Prop cauz while infering we 
        # used this function meaning, we dont want updated forward pass for NAG, while inferring.
        
        neuron_outputs = [data_X]
        # pass thorugh hidden layers
        if optimizer != 'NAG':
            for i in range(self.num_layers - 2): 
                
                # print("ran")
                # print(neuron_outputs[-1].shape)
                # print(self.weights[i].shape)
                # print(self.biases[i].shape)
                
                a = np.dot(neuron_outputs[-1], self.weights[i]) + self.biases[i]

                if activation_function == 'ReLu':
                    h = self.relu(a)
                elif activation_function == 'Sigmoid':
                    h = self.sigmoid(a)
                elif activation_function == 'tanh':
                    h = self.tanh(a)
                elif activation_function == 'Linear':
                    h = a
                else: 
                    raise ValueError("Error from forward pass: valid activation not passed!")
                neuron_outputs.append(h)
                
            # pass thorugh the output layer
            a = np.dot(neuron_outputs[-1], self.weights[-1]) + self.biases[-1]
            output = self.softmax(a)
            neuron_outputs.append(output)
            return neuron_outputs
            
        else: # if optimizer is NAG
            for i in range(self.num_layers - 2): 
                
                # print("ran")
                # print(neuron_outputs[-1].shape)
                # print(self.weights[i].shape)
                # print(self.biases[i].shape)
                
                a = np.dot(neuron_outputs[-1], self.weights[i] - self.beta_1 * self.weights_velocity[i]) 
                + self.biases[i] - self.beta_1 * self.bias_velocity[i] 
                
                if activation_function == 'ReLu':
                    h = self.relu(a)
                elif activation_function == 'Sigmoid':
                    h = self.sigmoid(a)
                elif activation_function == 'tanh':
                    h = self.tanh(a)
                elif activation_function == 'Linear':
                    h = a
                else: 
                    raise ValueError("Error from NAG forward pass - valid activation not passed!")
                    
                neuron_outputs.append(h)
                
            # pass thorugh the output layer
            a = np.dot(neuron_outputs[-1], self.weights[-1] - self.beta_1 * self.weights_velocity[-1]) 
            + self.biases[-1] - self.beta_1 * self.bias_velocity[-1]
            
            output = self.softmax(a)
            neuron_outputs.append(output)
            return neuron_outputs
            


    # def forward_pass_NAG(self, data_X):
    #     neuron_outputs = [data_X]
    #     # pass thorugh hidden layers

        
    #     for i in range(self.num_layers - 2): 
            
    #         # print("ran")
    #         # print(neuron_outputs[-1].shape)
    #         # print(self.weights[i].shape)
    #         # print(self.biases[i].shape)
            
    #         a = np.dot(neuron_outputs[-1], self.weights[i] - self.beta_1 * self.weights_velocity[i]) 
    #         + self.biases[i] - self.beta_1 * self.bias_velocity[i] 
            
    #         h = self.sigmoid(a)
    #         neuron_outputs.append(h)
            
    #     # pass thorugh the output layer
    #     a = np.dot(neuron_outputs[-1], self.weights[-1] - self.beta_1 * self.weights_velocity[-1]) 
    #     + self.biases[-1] - self.beta_1 * self.bias_velocity[-1]
        
    #     output = self.softmax(a)
    #     neuron_outputs.append(output)
    #     return neuron_outputs


    def sigmoid(self, X):
        # clipping it bw -500 to 500 
        return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1-x) 

    def relu(self, x):
        return np.maximum(0, x)

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
    
    def backword_pass(self, X, y, neuron_outputs, learning_rate, t, optimizer, loss_function, weight_decay, activation_function): 
        
        # compute the gradient at given value of params
        batch_size = len(X)
        
        # computing gradies with repect to the output layer
        if loss_function == 'categorical_cross_entropy':   
            delta = neuron_outputs[-1] - y # shape of delta is (batch_size x 10)
            
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
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) # since professor has not taught 
#weight decay in the class yet, by reading online i'm not encorporating weight decay for biases.
                
                if i>0: # because computing delta for i=0 will be absolutely un-necessary.
                    if activation_function == 'ReLu':
                        delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(neuron_outputs[i])
                    elif activation_function == 'Sigmoid': 
                        delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(neuron_outputs[i])
                    elif activation_function == 'tanh': 
                        delta = np.dot(delta, self.weights[i].T) * self.tanh_derivative(neuron_outputs[i])
                    elif activation_function == 'Linear': 
                        delta = np.dot(delta, self.weights[i].T) * np.ones_like(neuron_outputs[i])
                    else: 
                        raise ValueError("Error from backwork pass - pass valid activation function!")
                        
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
                    self.weights[i] -= dw * learning_rate / np.sqrt(self.history_weights[i] + 1e-12)
                    self.biases[i] -= db * learning_rate / np.sqrt(self.history_bias[i] + 1e-12)
    
                elif optimizer == 'Adam':
                    
                    # adding the momentum
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + (1 - self.beta_1) * dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + (1 - self.beta_1) * db
                    
                    # we need to change the learning rate at each iteration as per history accumulated for the current layer w and b
                    self.history_weights[i] = self.beta_2 * self.history_weights[i] + (1 - self.beta_2) * (dw ** 2)
                    self.history_bias[i] = self.beta_2 * self.history_bias[i] + (1 - self.beta_2) * (db ** 2) 
        
                    # implementing the adam rule for update with bias corrected
                    self.weights[i] -= self.weights_velocity[i]/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_weights[i] / (1 - self.beta_2** t) + 1e-12)
                    self.biases[i] -= self.bias_velocity[i]/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_bias[i] / (1 - self.beta_2 ** t) + 1e-12) 
    
                elif optimizer == 'Nadam':
                    
                    # adding the momentum
                    self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + (1 - self.beta_1) * dw
                    self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + (1 - self.beta_1) * db
        
                    # we need to change the learning rate at each iteration as per history accumulated for the current layer w&b
                    self.history_weights[i] = self.beta_2 * self.history_weights[i] + (1 - self.beta_2) * (dw ** 2)
                    self.history_bias[i] = self.beta_2 * self.history_bias[i] + (1 - self.beta_2) * (db ** 2) 
                    
                    # implementing the nadam rule for update with bias corrected
                    self.weights[i] -= ((self.beta_1 * self.weights_velocity[i]) + (1 - self.beta_1) * dw)/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_weights[i] / (1 - self.beta_2** t) + 1e-12)
                    self.biases[i] -= ((self.beta_1 * self.bias_velocity[i]) + (1 - self.beta_1) * db)/(1 - self.beta_1 ** t) * learning_rate / np.sqrt(self.history_bias[i] / (1 - self.beta_2 ** t) + 1e-12) 
                    
        else: # if optimizer is 'NAG'
            
            for i in range(self.num_layers - 2, -1, -1):
    
                # neuron_ouputs depend on the data, which we have computed in the forward pass
                # but delta will depend on on the weight matrices
                
                dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i] # however i am using regularied loss, and 
                # for computing the look ahead gradient value i have not used look ahead weight values!
                db = (np.sum(delta, axis=0, keepdims=True) / batch_size) 
                
                if i>0: # because computing delta for i=0 will be absolutely un-necessary.
                    
                    if activation_function == 'ReLu':
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * self.relu_derivative(neuron_outputs[i])
                    elif activation_function == 'Sigmoid': 
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * self.sigmoid_derivative(neuron_outputs[i])
                    elif activation_function == 'tanh': 
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * self.tanh_derivative(neuron_outputs[i])
                    elif activation_function == 'Linear': 
                        delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * np.ones_like(neuron_outputs[i])
                    else: 
                        raise ValueError('Error from backwork pass - pass valid activation function!')
                        
                # computing the momentum
                self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
                self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db
    
                # making an update
                self.weights[i] -= self.weights_velocity[i] * learning_rate
                self.biases[i] -= self.bias_velocity[i] * learning_rate
    
    # # if optmizer is NAG
    # def backword_pass_NAG(self, X, y, neuron_outputs, learning_rate, weight_decay):
        
    #     # compute the gradient at given value of params
    #     batch_size = len(X)
        
    #     # Computing gradies with repect to the output layer 
    #     delta = neuron_outputs[-1] - y
        
    #     # computing gradient with respect to hidden layers
    #     for i in range(self.num_layers - 2, -1, -1):

    #         # neuron_ouputs depend on the data, which we have computed in the forward pass
    #         # but delta will depend on on the weight matrices
            
    #         dw = (np.dot(neuron_outputs[i].T, delta) / batch_size) + weight_decay * self.weights[i]
    #         db = (np.sum(delta, axis=0, keepdims=True) / batch_size) + weight_decay * self.biases[i]
            
    #         if i>0: # because computing delta for i=0 will be absolutely un-necessary.
    #             delta = np.dot(delta, (self.weights[i] - self.beta_1 * self.weights_velocity[i]).T) * self.sigmoid_derivative(neuron_outputs[i])

    #         # computing the momentum
    #         self.weights_velocity[i] = self.beta_1 * self.weights_velocity[i] + dw
    #         self.bias_velocity[i] = self.beta_1 * self.bias_velocity[i] + db

    #         # making an update
    #         self.weights[i] -= self.weights_velocity[i] * learning_rate
    #         self.biases[i] -= self.bias_velocity[i] * learning_rate
        
    # function to train the network
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001, batch_size=32, optimizer="RMS_Prop", beta_1=0.900, beta_2=0.999, loss_function='categorical_cross_entropy', weight_decay = 0, activation_function='ReLu'): 
        # setting RMS_Prop as default optimizer

        # we could've set 
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        activation_function = self.activation_function
        
        global_step = 0
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
                neuron_outputs = self.forward_pass(batch_x, optimizer, activation_function)

                l2_norm_weights = 0
                for i in range(len(self.weights)):
                    l2_norm_weights += np.sum(self.weights[i] ** 2)              
                                      
                if loss_function == 'categorical_cross_entropy':
                    loss = -np.mean(np.sum(batch_y * np.log(neuron_outputs[-1] + 1e-10), axis = 1)) +  (weight_decay/2) * l2_norm_weights 
                    # added 1e-10 to prevent num underflow
                else:
                    loss = np.mean(np.sum((batch_y - neuron_outputs[-1]) ** 2, axis = 1)) + (weight_decay/2) * l2_norm_weights
                    # loss function for MSE
            
                total_loss += loss
                batch_num += 1
                global_step += 1
                # backprop: computing the gradient and making an update in backwork pass func
                self.backword_pass(batch_x, batch_y, neuron_outputs, learning_rate, global_step, optimizer, loss_function, weight_decay, activation_function)
            
            average_loss = total_loss/batch_num
            
            # now let's make predictions on validation dataset 
            validation_predictions = self.predict(X_val)
            validation_accuracy = np.mean(validation_predictions == np.argmax(y_val, axis = 1))

            # loging to wandb
            wandb.log({'epoch': epoch , 'train_loss': average_loss, 'val_accuracy': validation_accuracy})
            print(f"epoch: {epoch}, train_loss:{average_loss:.4f}, val_accuracy: {validation_accuracy:.4f}")
            
            
    def predict(self, X): 
        # this will predict class labels for the passed data
        neuron_outputs = self.forward_pass(X)
        return np.argmax(neuron_outputs[-1], axis=1)


if __name__ == "__main__": # will run only when script will be executed directly
    np.random.seed(42) # to get teh same results again
    # will check first which data is given in the arguement, yet to do that
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # making the data ready to train the model

    # Splitting the trainig data into train and validation
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    train_size = 50000

    train_x = train_images[indices[:train_size]] / 255
    train_y = train_labels[indices[:train_size]] 
    val_x = train_images[indices[train_size:]] / 255
    val_y = train_labels[indices[train_size:]]

    train_x = train_x.reshape(train_x.shape[0], -1)
    val_x = val_x.reshape(val_x.shape[0], -1)

    # converting y's into one hot vector
    num_classes = 10
    train_y = np.eye(num_classes)[train_y]
    val_y = np.eye(num_classes)[val_y]

    # let's do it for test data as well
    test_images = test_images.reshape(test_images.shape[0], -1) / 255
    test_labels = np.eye(num_classes)[test_labels]

    # initializing hyperparms in wandb
    optimizers = ['sgd', 'momentum sgd','NAG','RMS_Prop','Adam','Nadam']
    loss = ['mean_squared', 'categorical_cross_entropy']
    activation_function = ['ReLu', 'Sigmoid', 'Linear', 'tanh']
    initialization = ['Xavier', 'He']
    
    wandb.init(project='backprop_scratch', 
            config={ 'Learning_rate' : 0.0001, 
                        'epochs' : 10, 
                        'batch_size' : 32, 
                        'layer_size' : [784, 128, 64, 10], 
                    'optimizer': 'RMS_Prop',
                    'beta_1': 0.900,         # beta 1 is for momentum sgd, NAG, RMS_Prop
                    'beta_2': 0.999,         # use beta 2 for Nadam and Adam
                   'loss_function': 'mean_squared',
                    'weight_decay': 0,
                    'activation_function': 'ReLu',
                    'initialization': 'He'
                   }         
            ) # select optimizer from optimizer list up there 
    
    config = wandb.config
    config.beta_1 = 0.900
    config.beta_2 = 0.999
    config.loss_function = 'categorical_cross_entropy'
    config.weight_decay = 0
    config.optimizer = 'Adam'
    config.activation_function = 'Sigmoid'
    config.Learning_rate = 0.001
    config.batch_size = 32 
    config.epochs = 10
    print(wandb.config)

    # now let's create and train the network
    model = backprop_from_scratch(config.layer_size, config.activation_function)
    model.train(train_x, train_y, val_x, val_y, config.epochs, config.Learning_rate, config.batch_size, config.optimizer, config.beta_1, config.beta_2, config.loss_function, config.weight_decay, config.activation_function)
    # now let's evaluate on the test set
    test_predictions = model.predict(test_images) 
    test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis = 1))
    print(f"test_accuracy:{test_accuracy: .4f}") 
    wandb.log({'test_accuracy': test_accuracy})

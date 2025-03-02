import numpy as np 

# defining a backprop_from_scratch to do all.
class backprop_from_scratch:

    # define constructor
    def __init__(self, layer_size):
        # initialise the neural network
        
        self.weights, self.biases = [], []
        self.num_layers = len(layer_size)
        self.layer_sizes = layer_size
        self.initialize_params()
        
    def initialize_params(self):
        # let's use He initialization for weights and set values of bias equals to zero
        for i in range(self.num_layers-1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2/self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w) 
            self.biases.append(b) 
            
    def forward_pass(self, data_X):
        neuron_outputs = [data_X]
        # pass thorugh hidden layers
        for i in range(self.num_layers - 2): 
            
            # print("ran")
            # print(neuron_outputs[-1].shape)
            # print(self.weights[i].shape)
            # print(self.biases[i].shape)
            
            a = np.dot(neuron_outputs[-1], self.weights[i]) + self.biases[i]
            h = self.sigmoid(a)
            neuron_outputs.append(h)
        # pass thorugh the output layer
        a = np.dot(neuron_outputs[-1], self.weights[-1]) + self.biases[-1]
        output = self.softmax(a)
        neuron_outputs.append(output)
        return neuron_outputs

    def sigmoid(self, X):
        # clipping it bw -500 to 500 
        return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x*(1-x) 
        
    def softmax(self, X): 
        # clipping it for numerical stability
        exp_x = np.exp(X - np.max(X, axis = 1, keepdims=True))
        return exp_x/np.sum(exp_x, axis = 1, keepdims=True)

    def compute_gradients(self, neuron_outputs, delta, batch_size):
        dw = np.dot(neuron_outputs[i].T, delta) / batch_size
        db = np.sum(delta, axis=0, keepdims=True) / batch_size
        return dw, db

    def make_an_update(dw, db, learning_rate):
        self.weights[i] -= dw * learning_rate
        self.biases[i] -= db * learning_rate
        
    def backword_pass(self, X, y, neuron_outputs, learning_rate):

        # compute the gradient at given value of params
        batch_size = len(X)
        # computing gradies with repect to the output layer 
        delta = neuron_outputs[-1] - y
        # computing gradient with respect to hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            
            dw, db = compute_gradients(neuron_outputs[i].T, delta, batch_size)
            
            # dw = np.dot(neuron_outputs[i].T, delta) / batch_size
            # db = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            if i > 0: # because computing delta for i=0 will be absolutely un-necessary.
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(neuron_outputs[i])

            make_an_update(dw, db, learning_rate)
            
            # self.weights[i] -= dw * learning_rate
            # self.biases[i] -= db * learning_rate
        
    # function to train the network
    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size):
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
                neuron_outputs = self.forward_pass(batch_x)
                loss = -np.mean(np.sum(batch_y * np.log(neuron_outputs[-1] + 1e-10), axis = 1)) # to prevent numberical underflow
                total_loss += loss
                batch_num += 1
                self.backword_pass(batch_x, batch_y, neuron_outputs, learning_rate)
            
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
    
# Splitting the trainig data into train and validation

indices = np.arange(train_images.shape[0])
np.random.shuffle(indices)
train_size = 50000
train_x = train_images[indices[:train_size]]
train_y = train_labels[indices[:train_size]]
val_x = train_images[indices[train_size:]]
val_y = train_labels[indices[train_size:]]

train_x = train_x.reshape(train_x.shape[0], -1)
val_x = val_x.reshape(val_x.shape[0], -1)

# train_x.ravel()
# val_y.ravel()

# converting y's into one hot vector
num_classes = 10
train_y = np.eye(num_classes)[train_y]
val_y = np.eye(num_classes)[val_y]

# let's do it for test data as well
test_images = test_images.reshape(test_images.shape[0], -1)
test_labels = np.eye(num_classes)[test_labels]
# initializing hyperparms in wandb
wandb.init(project='backprop_scratch', 
           config={ 'Learning_rate' : 0.001, 
                    'epochs' : 50, 
                    'batch_size' : 32, 
                    'layer_size' : [784, 128, 64, 10]})

print(wandb.config)




config = wandb.config

# now let's create and train the network
model = backprop_from_scratch(config.layer_size)

model.train(train_x, train_y, val_x, val_y, config.epochs, config.Learning_rate, config.batch_size)

# now let's evaluate on the test set
test_predictions = model.predict(test_images) 
test_accuracy = np.mean(test_predictions == np.argmax(test_labels, axis = 1))
print(f"test_accuracy:{test_accuracy: .4f}") 
wandb.log({'test_accuracy': test_accuracy})

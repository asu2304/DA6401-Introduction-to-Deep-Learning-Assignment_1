# Original
def train(self, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001, batch_size=32, optimizer="RMS_Prop", ...):
    for epoch in range(epochs):
        batch_num = 0
        for i in range(0, X_train.shape[0], batch_size):
            batch_num += 1
            self.backword_pass(batch_x, batch_y, neuron_outputs, learning_rate, batch_num, optimizer, ...)

# Corrected
def train(self, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001, batch_size=32, optimizer="RMS_Prop", ...):
    global_step = 0
    for epoch in range(epochs):
        batch_num = 0
        for i in range(0, X_train.shape[0], batch_size):
            batch_num += 1
            global_step += 1
            self.backward_pass(batch_x, batch_y, neuron_outputs, learning_rate, global_step, optimizer, ...)
import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = np.zeros((1, 1))
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
    
    def backward(self, inputs, gradient):
        self.weights_grad = np.dot(inputs.T, gradient)
        self.bias_grad = np.sum(gradient, axis=0, keepdims=True)
        return np.dot(gradient, self.weights.T)

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(inputs)
    
    def backward(self, gradient):
        return gradient * self.activation_prime(self.inputs)

class Layer:
    def __init__(self, input_size, output_size, activation, activation_prime):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]
        self.activation = ActivationLayer(activation, activation_prime)
        
    def forward(self, inputs):
        self.inputs = inputs
        neuron_outputs = np.hstack([neuron.forward(inputs) for neuron in self.neurons])
        return self.activation.forward(neuron_outputs)
    
    def backward(self, gradient):
        act_gradient = self.activation.backward(gradient)
        neuron_gradients = [neuron.backward(self.inputs, act_gradient[:, i:i+1]) for i, neuron in enumerate(self.neurons)]
        return np.sum(neuron_gradients, axis=0)
    
    def update(self, learning_rate):
        for neuron in self.neurons:
            neuron.weights -= learning_rate * neuron.weights_grad
            neuron.bias -= learning_rate * neuron.bias_grad

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def add_layer(self, input_size, output_size, activation, activation_prime):
        self.layers.append(Layer(input_size, output_size, activation, activation_prime))
        
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
            
    def train(self, X, y, epochs, learning_rate, batch_size=4):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = np.mean((y_pred - y_batch)**2)
                total_loss += loss
                
                # Backward pass
                gradient = 2 * (y_pred - y_batch) / batch_size
                self.backward(gradient)
                
                # Update weights
                self.update(learning_rate)
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(X)*batch_size}')

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the network
nn = NeuralNetwork()
nn.add_layer(2, 4, sigmoid, sigmoid_prime)
nn.add_layer(4, 1, sigmoid, sigmoid_prime)

nn.train(X, y, epochs=10000, learning_rate=0.05, batch_size=100)

# Test the network
for inputs in X:
    prediction = nn.forward(inputs.reshape(1, -1))
    print(f"Input: {inputs}, Prediction: {prediction[0][0]:.4f}")
# nodes.py

import numpy as np

class NeuralNode:
    def __init__(self, name, activation_function=None, is_input=False, is_output=False):
        self.name = name
        self.activation_function = activation_function
        self.is_input = is_input
        self.is_output = is_output
        self.incoming_nodes = []
        self.weights = None
        self.bias = np.random.randn()
        self.value = None
        self.gradient = None

    def add_incoming_node(self, node):
        self.incoming_nodes.append(node)
        if self.weights is None:
            self.weights = np.random.randn(len(self.incoming_nodes))
        else:
            self.weights = np.random.randn(len(self.incoming_nodes))

    def forward(self):
        if self.is_input:
            return self.value
        
        incoming_values = np.array([node.forward() for node in self.incoming_nodes])
        self.value = np.dot(self.weights, incoming_values) + self.bias
        
        if self.activation_function:
            self.value = self.activation_function(self.value)
        
        return self.value

    def backward(self, upstream_gradient, learning_rate=0.1):
        local_gradient = 1
        if self.activation_function:
            local_gradient = self.activation_function(self.value, derivative=True)
        
        self.gradient = upstream_gradient * local_gradient
        
        for i, node in enumerate(self.incoming_nodes):
            node.backward(self.gradient * self.weights[i], learning_rate)
            self.weights[i] -= learning_rate * self.gradient * node.value
        
        self.bias -= learning_rate * self.gradient

    def reset(self):
        self.value = None
        self.gradient = None

    def __repr__(self):
        return f"NeuralNode(name={self.name}, value={self.value}, gradient={self.gradient}, weights={self.weights}, bias={self.bias})"

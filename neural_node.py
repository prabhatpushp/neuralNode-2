import numpy as np

class NeuralNode:
    def __init__(self, name, activation_function=None, is_input=False, is_output=False):
        self.name = name
        self.activation_function = activation_function
        self.is_input = is_input
        self.is_output = is_output
        self.incoming_nodes = []
        self.outgoing_nodes = []
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

    def backward(self, upstream_gradient):
        local_gradient = 1
        if self.activation_function:
            local_gradient = self.activation_function(self.value, derivative=True)
        
        self.gradient = upstream_gradient * local_gradient
        
        for i, node in enumerate(self.incoming_nodes):
            node.backward(self.gradient * self.weights[i])
            self.weights[i] -= 0.1 * self.gradient * node.value
        
        self.bias -= 0.1 * self.gradient

    def __repr__(self):
        return f"NeuralNode(name={self.name}, value={self.value}, gradient={self.gradient}, weights={self.weights}, bias={self.bias})"

def sigmoid(x, derivative=False):
    if derivative:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-x))

input_node1 = NeuralNode(name='Input1', is_input=True)
input_node2 = NeuralNode(name='Input2', is_input=True)
hidden_node1 = NeuralNode(name='Hidden1', activation_function=sigmoid)
hidden_node2 = NeuralNode(name='Hidden2', activation_function=sigmoid)
hidden_node3 = NeuralNode(name='Hidden3', activation_function=sigmoid)
hidden_node4 = NeuralNode(name='Hidden4', activation_function=sigmoid)
output_node1 = NeuralNode(name='Output1', activation_function=sigmoid, is_output=True)
output_node2 = NeuralNode(name='Output2', activation_function=sigmoid, is_output=True)

hidden_node1.add_incoming_node(input_node1)
hidden_node1.add_incoming_node(input_node2)

hidden_node2.add_incoming_node(input_node1)
hidden_node2.add_incoming_node(input_node2)

hidden_node3.add_incoming_node(input_node1)
hidden_node3.add_incoming_node(input_node2)

hidden_node4.add_incoming_node(input_node1)
hidden_node4.add_incoming_node(input_node2)

output_node1.add_incoming_node(hidden_node1)
output_node1.add_incoming_node(hidden_node2)
output_node1.add_incoming_node(hidden_node3)
output_node1.add_incoming_node(hidden_node4)

output_node2.add_incoming_node(hidden_node1)
output_node2.add_incoming_node(hidden_node2)
output_node2.add_incoming_node(hidden_node3)
output_node2.add_incoming_node(hidden_node4)

xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs1 = [0, 1, 1, 0]
xor_outputs2 = [1, 0, 0, 1]  # Assuming another output with opposite labels

learning_rate = 0.09
for epoch in range(100000):
    total_loss = 0
    for (x1, x2), y1, y2 in zip(xor_inputs, xor_outputs1, xor_outputs2):
        input_node1.value = x1
        input_node2.value = x2
        predicted_output1 = output_node1.forward()
        predicted_output2 = output_node2.forward()
        
        loss1 = (predicted_output1 - y1) ** 2
        # loss2 = (predicted_output2 - y2) ** 2
        loss2 = 0
        total_loss += loss1 + loss2
        
        output_node1.backward(2 * (predicted_output1 - y1))
        output_node2.backward(2 * (predicted_output2 - y2))
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')

for (x1, x2), y1, y2 in zip(xor_inputs, xor_outputs1, xor_outputs2):
    input_node1.value = x1
    input_node2.value = x2
    predicted_output1 = output_node1.forward()
    predicted_output2 = output_node2.forward()
    print(f"Input: ({x1}, {x2}), Predicted Output1: {predicted_output1}, Expected Output1: {y1}")
    print(f"Input: ({x1}, {x2}), Predicted Output2: {predicted_output2}, Expected Output2: {y2}")

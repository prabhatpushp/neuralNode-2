# xor_example.py

import numpy as np
from neural_network import NeuralNode, sigmoid, relu, tanh, MeanSquaredError

# Define the network structure
input_node1 = NeuralNode(name='Input1', is_input=True)
input_node2 = NeuralNode(name='Input2', is_input=True)

hidden_node1 = NeuralNode(name='Hidden1', activation_function=relu)
hidden_node2 = NeuralNode(name='Hidden2', activation_function=relu)
hidden_node3 = NeuralNode(name='Hidden3', activation_function=relu)
hidden_node4 = NeuralNode(name='Hidden4', activation_function=relu)
hidden_node5 = NeuralNode(name='Hidden5', activation_function=relu)
hidden_node6 = NeuralNode(name='Hidden6', activation_function=relu)

output_node1 = NeuralNode(name='Output1', activation_function=sigmoid, is_output=True)
output_node2 = NeuralNode(name='Output2', activation_function=sigmoid, is_output=True)

# Connect the nodes
hidden_node1.add_incoming_node(input_node1)
hidden_node1.add_incoming_node(input_node2)

hidden_node2.add_incoming_node(input_node1)
hidden_node2.add_incoming_node(input_node2)

hidden_node3.add_incoming_node(hidden_node1)
hidden_node3.add_incoming_node(hidden_node2)

hidden_node4.add_incoming_node(hidden_node1)
hidden_node4.add_incoming_node(hidden_node2)

hidden_node5.add_incoming_node(hidden_node3)
hidden_node5.add_incoming_node(hidden_node4)

hidden_node6.add_incoming_node(hidden_node3)
hidden_node6.add_incoming_node(hidden_node4)

output_node1.add_incoming_node(hidden_node5)
output_node1.add_incoming_node(hidden_node6)

output_node2.add_incoming_node(hidden_node5)
output_node2.add_incoming_node(hidden_node6)

# XOR inputs and expected outputs
xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs1 = [0, 1, 1, 0]
xor_outputs2 = [1, 0, 0, 1]  # Assuming another output with opposite labels

# Training the network
loss_function = MeanSquaredError()
learning_rate = 0.1

for epoch in range(100000):
    total_loss = 0
    for (x1, x2), y1, y2 in zip(xor_inputs, xor_outputs1, xor_outputs2):
        input_node1.value = x1
        input_node2.value = x2
        
        # Forward pass
        predicted_output1 = output_node1.forward()
        predicted_output2 = output_node2.forward()
        
        # Calculate loss
        loss1 = loss_function.loss(predicted_output1, y1)
        loss2 = loss_function.loss(predicted_output2, y2)
        total_loss += loss1 + loss2
        
        # Backward pass
        grad1 = loss_function.gradient(predicted_output1, y1)
        grad2 = loss_function.gradient(predicted_output2, y2)
        
        output_node1.backward(grad1, learning_rate)
        output_node2.backward(grad2, learning_rate)
        
        # Reset node values and gradients
        # for node in [input_node1, input_node2, hidden_node1, hidden_node2, hidden_node3, hidden_node4, hidden_node5, hidden_node6, output_node1, output_node2]:
        #     node.reset()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')

# Testing the network
for (x1, x2), y1, y2 in zip(xor_inputs, xor_outputs1, xor_outputs2):
    input_node1.value = x1
    input_node2.value = x2
    predicted_output1 = output_node1.forward()
    predicted_output2 = output_node2.forward()
    print(f"Input: ({x1}, {x2}), Predicted Output1: {predicted_output1}, Expected Output1: {y1}")
    print(f"Input: ({x1}, {x2}), Predicted Output2: {predicted_output2}, Expected Output2: {y2}")

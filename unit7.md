Unit 7: Perceptron Activities

After running the three Jupyter Notebook files provided in this unit's formative acitivity I wrote some python code to understand the concept of perceptrons better.

Here is the code I wrote:

import numpy as np

# Perceptron Functions
def dot_product(inputs, weights):
    return np.dot(inputs, weights)

def step_function(value):
    return 1 if value >= 1 else 0

def perceptron_output(instance, weights):
    return step_function(dot_product(instance, weights))

def train_perceptron(inputs, outputs, weights, learning_rate):
    while True:
        total_error = 0
        for i, target in enumerate(outputs):
            prediction = perceptron_output(inputs[i], weights)
            error = target - prediction
            total_error += abs(error)
            if error != 0:
                weights += learning_rate * inputs[i] * error
        if total_error == 0:
            break
    return weights

# Activation Functions
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoid_derivative(value):
    return value * (1 - value)

# Exercise 1: Simple Perceptron Example
inputs1 = np.array([45, 25])
weights1 = np.array([0.7, 0.1])
step_result1 = step_function(dot_product(inputs1, weights1))

weights1 = np.array([-0.7, 0.1])
step_result2 = step_function(dot_product(inputs1, weights1))

# Exercise 2: Perceptron Training
inputs2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs2 = np.array([0, 0, 0, 1])
weights2 = np.zeros(2)
learning_rate = 0.1

trained_weights = train_perceptron(inputs2, outputs2, weights2, learning_rate)

# Exercise 3: Multi-Layer Neural Network
inputs3 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs3 = np.array([[0], [1], [1], [0]])

# Initialize weights
weights_0 = np.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
weights_1 = np.array([[-0.017], [-0.893], [0.148]])

# Forward Pass
hidden_layer = sigmoid(np.dot(inputs3, weights_0))
output_layer = sigmoid(np.dot(hidden_layer, weights_1))

# Error Calculation
error_output_layer = outputs3 - output_layer
average_error = np.mean(abs(error_output_layer))

# Backpropagation
output_delta = error_output_layer * sigmoid_derivative(output_layer)
hidden_delta = np.dot(output_delta, weights_1.T) * sigmoid_derivative(hidden_layer)

# Weight Updates
weights_1 += np.dot(hidden_layer.T, output_delta) * learning_rate
weights_0 += np.dot(inputs3.T, hidden_delta) * learning_rate


Exercise 1 demonstrates the behavior of a simple perceptron. The perceptron calculates a weighted sum of inputs and passes it through a step function to produce a binary output. The purpose here is to show how changes in weights can change the decision boundary of the perceptron. I am providing two sets of weights for the same input which allows me to observe how the perceptron's output changes. Even slight changes in weights offer different outcomes.

In exercise 2 the concept from the previous exercise is expanded by introducing a training mechanism for the perceptron. Here I implemented and trained a perceptron to model a logical AND gate. The perceptron learns through a process of iteratively adjusting its weights based on errors between predicted and target outputs. This helped me understand a perceptron’s ability to learn linearly separable functions. This attempt failed for me a few times, however by researching online, I was able to find an explanation by Dukor (2018) that helped me understand the concept better and allowed me to implement this properly.


In the third exercise I created a very simple multi-layer neural network capable of solving more complex problems, such as modeling an XOR gate. The forward pass computes predictions by transforming inputs through layers of neurons using the sigmoid activation function. The backward pass works by taking the mistakes the neural network makes in its predictions and using them to adjust the connections between the layers. This adjustment helps the network improve its accuracy over time by learning from its errors.

References:

Dukor, O.S., 2018. Neural Representation of AND, OR, NOT, XOR and XNOR Logic Gates (Perceptron Algorithm). Medium. Available at: https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1 [Accessed 2 January 2025].
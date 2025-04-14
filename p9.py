# CREATE VERTICAL DATA WITH 3 CLASSES

import matplotlib.pyplot as plt
# from nnfs.datasets.vertical import create_data
from nnfs.datasets.spiral import create_data
X, y = create_data(samples=100, classes=3)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

# MAKE THE NETW0RK

from classes import Layer_Dense,Activation_ReLU,Activation_Softmax,Loss_CategoricalCrossEntropy
import numpy as np

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

# Just using random numbers:
def optimize_rand(dense1, dense2, loss_function, X, y, iterations, adj_k):
    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    loss_record = np.zeros(iterations)

    for iteration in range(iterations):
        dense1.weights = adj_k * np.random.randn(2,3)
        dense1.biases = adj_k * np.random.randn(1,3)
        dense2.weights = adj_k * np.random.randn(3,3)
        dense2.biases = adj_k * np.random.randn(1,3)
        
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output, y)
        
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions==y)

        if loss < lowest_loss:
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            print('New set of weights found:, iteration:', iteration,
                  'loss:', lowest_loss, 'acc:', accuracy)
        loss_record[iteration] = lowest_loss
    plt.scatter(range(iterations),loss_record)
    plt.show()

# Using random adjustments:
def optimize_walk(dense1, dense2, loss_function, X, y, iterations, adj_k):
    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    loss_record = np.zeros(iterations)

    for iteration in range(iterations):
        dense1.weights += adj_k * np.random.randn(2,3)
        dense1.biases += adj_k * np.random.randn(1,3)
        dense2.weights += adj_k * np.random.randn(3,3)
        dense2.biases += adj_k * np.random.randn(1,3)
        
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output, y)
        
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions==y)

        if loss < lowest_loss:
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            print('New set of weights found:, iteration:', iteration,
                  'loss:', lowest_loss, 'acc:', accuracy)
        else: 
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()
        loss_record[iteration] = lowest_loss

    plt.scatter(range(iterations),loss_record)
    plt.show()


iterations = 100000

# optimize_rand(dense1, dense2, loss_function, X, y, iterations, .05)
optimize_walk(dense1, dense2, loss_function, X, y, iterations, .05)

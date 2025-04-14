import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -.91, .26, -0.5],
           [-0.26, -0.27, .17, .87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -.14, .5],
           [-0.5, .12, -.33],
           [-.44, .73, -.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)


import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -.91, .26, -0.5],
           [0.26, -0.27, .17, .87]]

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)
'''
# loop through neurons
for neuron_weights, neuron_bias in zip(weights, biases):
    # initialize output
    neuron_output = 0

    # loop through input, weight pairs
    for n_input, weight in zip(inputs, neuron_weights):
        # add to output : input * weight
        neuron_output += n_input * weight

    # add the bias
    neuron_output += neuron_bias

    # append the neuron output
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''

a = [1,2,3]
b = [2,3,4]


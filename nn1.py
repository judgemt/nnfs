import numpy as np
from nnfs.datasets.spiral import create_data
import matplotlib.pyplot as plt
from classes import Layer_Dense,Activation_ReLU,Activation_Softmax,Loss_CategoricalCrossEntropy

X, y = create_data(100, 3)

# plt.scatter(X[:,0], X[:,1])
# plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)

np.argmax()
import numpy as np
from nnfs.datasets.spiral import create_data

import matplotlib.pyplot as plt

X, y = create_data(100, 3)

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()

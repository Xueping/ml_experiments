import numpy as np
from keras.datasets import mnist

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
np.random.seed(123)  # for reproducibility
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
# (60000, 28, 28)

plt.plot(X_train[0])
plt.show()

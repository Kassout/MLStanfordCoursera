import numpy as np


def sigmoid(z):
    """
        sigmoid() Compute sigmoid function
        g = sigmoid(z) computes the sigmoid of z.
    """

    # You need to return the following variables correctly
    g = np.zeros(np.shape(z))

    # Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    g = 1 / (1 + np.exp(-z))

    return g
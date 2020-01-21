import numpy as np


def computeCost(X, y, theta):
    # Initialize some useful values
    # number of training examples
    m = len(y)

    # You need to return the following variables correctly
    J = 0

    J = 1 / (2*m) * np.sum(np.square(np.subtract(np.matmul(X, theta).reshape(len(X), 1), y)))

    return J

import numpy as np


def computeCostMulti(X, y, theta):
    """
        computeCostMulti() Compute cost for linear regression with multiple variables
        J = computeCostMulti(X, y, theta) computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y
    """

    # Initialize some useful values
    # number of training examples
    m = len(y)

    # You need to return the following variables correctly
    J = 0

    # Compute the cost of a particular choice of theta
    J = 1 / (2 * m) * np.sum(np.square(np.subtract(np.matmul(X, theta), y)))

    return J
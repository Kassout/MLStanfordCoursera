import numpy as np


def normalEqn(X, y):
    """
        normalEqn() Computes the closed-form solution to linear regression
        normalEqn(X,y) computes the closed-form solution to linear
        regression using the normal equations.
    """

    theta = np.zeros((np.shape(X)[1], 1))

    theta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    return theta
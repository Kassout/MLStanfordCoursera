import numpy as np


def mapFeature(X1, X2):
    """
        mapFeature() Feature mapping function to polynomial features

        mapFeature(X1, X2) maps the two input features
        to quadratic features used in the regularization exercise.

        Returns a new feature array with more features, comprising of
        X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

        Inputs X1, X2 must be the same size
    """
    
    degree = 7
    if np.shape(X1) == ():
        out = np.ones((1, 28))
    else:
        out = np.ones((np.shape(X1)[0], 28))
    index = 0
    for i in range(degree):
        for j in range(i+1):
            out[:, index] = np.transpose(np.power(X1, (i - j)) * np.power(X2, j))
            index += 1

    return out
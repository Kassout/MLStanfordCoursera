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
    
    degree = 6
    out = np.ones((1, degree))
    for i in range(degree):
        for j in range(i+1):
            out[:, len(out)+1] = (X1 ^ (i - j)) * (X2 ^ j)
    
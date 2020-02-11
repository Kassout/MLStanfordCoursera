from python.sigmoid import sigmoid
import numpy as np


def predict(theta, X):
    """
        predict() Predict whether the label is 0 or 1 using learned logistic
        regression parameters theta
        p = predict(theta, X) computes the predictions for X using a
        threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

    # Number of training examples
    m = len(X)

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    p = sigmoid(X @ theta) >= 0.5

    return p

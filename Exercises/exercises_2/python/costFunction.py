import numpy as np

from python.sigmoid import sigmoid


def costFunction(theta, X, y):
    """
        costFunction() Compute cost for logistic regression
        J = costFunction(theta, X, y) computes the cost of using theta as the
        parameter for logistic regression
    """

    # Compute the cost of a particular choice of theta.
    # You should set J to the cost.
    J = (1 / len(y)) * (-np.transpose(y) @ np.log(sigmoid(X @ theta)) - np.transpose(1-y) @ np.log(1-sigmoid(X @ theta)))

    return J


def computeGrad(theta, X, y):
    """
        computeGrad() Compute gradient for logistic regression
        grad = computeGrad(theta, X, y) computes the gradient of the cost
        w.r.t. to the parameters.
    """
    # Compute the partial derivatives and set grad to the partial
    # derivatives of the cost w.r.t. each parameter in theta
    # Note: grad should have the same dimensions as theta
    grad = (1 / len(y)) * np.transpose(X) @ (sigmoid(X @ theta) - y)

    return grad
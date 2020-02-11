import numpy as np

from python.sigmoid import sigmoid


def costFunctionReg(theta, X, y, lmbda):
    """
        costFunctionReg() Compute cost for logistic regression with regulation
        J = costFunctionReg(theta, X, y) computes the cost of using theta as the
        parameter for logistic regression
    """

    # Compute the cost of a particular choice of theta.
    # You should set J to the cost.
    J = (1 / len(y)) * (-np.transpose(y) @ np.log(sigmoid(X @ theta)) - np.transpose(1-y) @ np.log(1-sigmoid(X @ theta))) + (lmbda / (2*len(y))) * (np.transpose(theta[1:len(theta)]) @ theta[1:len(theta)])

    return J


def computeGradReg(theta, X, y, lmbda):
    """
        computeGrad() Compute gradient for logistic regression with regulation
        grad = computeGrad(theta, X, y) computes the gradient of the cost
        w.r.t. to the parameters.
    """
    # Compute the partial derivatives and set grad to the partial
    # derivatives of the cost w.r.t. each parameter in theta
    # Note: grad should have the same dimensions as theta
    grad = np.zeros((len(theta), 1))

    grad[0] = (1 / len(y)) * np.transpose(X[:, 0]) @ (sigmoid(X @ theta.reshape((len(theta), 1))) - y)
    grad[1:len(grad)] = (1 / len(y)) * np.transpose(X[:, 1:len(X)]) @ (sigmoid(X @ theta.reshape((len(theta), 1))) - y) + (lmbda / len(y)) * theta[1:len(theta)].reshape((len(theta)-1, 1))

    return grad
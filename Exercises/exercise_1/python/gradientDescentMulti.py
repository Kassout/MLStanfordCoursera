import numpy as np

from python.computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
        gradientDescentMulti() Performs gradient descent to learn theta
        theta = gradientDescentMulti(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    # number of training examples
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):

        # Perform a single gradient step on the parameter vector theta.

        theta = np.subtract(theta, alpha / m * (np.transpose(X)@(np.subtract(np.matmul(X, theta), y))))

        # Save the cost J in every iteration
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history
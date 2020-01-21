import numpy as np

from python.computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):

    # Initialize some useful values
    # number of training examples
    m = len(y)

    J_history = np.zeros((num_iters, 1))

    for iter in range(1, num_iters):

        theta = theta - (alpha*(1/m) * sum(np.multiply(np.subtract(np.matmul(X, theta).reshape(len(X), 1), y), X))).reshape(len(theta), 1)

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return theta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize
from pip._vendor.distlib.compat import raw_input

from python.costFunction import costFunction, computeGrad
from python.costFunctionReg import costFunctionReg, computeGradReg
from python.mapFeature import mapFeature
from python.plotData import plotData
from python.plotDecisionBoundary import plotDecisionBoundary
from python.predict import predict

"""
    Machine Learning Online Class - Exercise 2: Logistic Regression

    Instructions
    ------------

    This file contains code that helps you get started on the second part
    of the exercise which covers regularization with logistic regression.

    You will need to complete the following functions in this exericse:

    sigmoid.m
    costFunction.m
    predict.m
    costFunctionReg.m

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
"""


def run():

    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).

    data = np.genfromtxt('ex2data2.csv', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, [2]]

    plotData(X, y)
    
    # Put some labels
    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    # Specified in plot order
    plt.legend(('y = 1', 'y = 0'))
    plt.show()
    
    # =========== Part 1: Regularized Logistic Regression ============
    # In this part, you are given a dataset with data points that are not
    # linearly separable. However, you would still like to use logistic
    # regression to classify the data points.
    # 
    # To do so, you introduce more features to use -- in particular, you add
    # polynomial features to our data matrix (similar to polynomial
    # regression).

    # Add Polynomial Features
    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros((np.shape(X)[1], 1))

    # Set regularization parameter lambda to 1
    lmbda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lmbda), computeGradReg(initial_theta, X, y, lmbda)

    print('Cost at initial theta (zeros): ' + str(cost) + '\n')
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros) - first five values only:\n')
    print(grad[0:5], '\n')
    print('Expected gradients (approx) - first five values only:\n')
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

    print('\nProgram paused. Press enter to continue.\n')
    raw_input()

    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones((np.shape(X)[1], 1))
    cost, grad = costFunctionReg(test_theta, X, y, 10), computeGradReg(test_theta, X, y, 10)

    print('\nCost at test theta (with lambda = 10): ' + str(cost) + '\n')
    print('Expected cost (approx): 3.16\n')
    print('Gradient at test theta - first five values only:\n')
    print(grad[0:5], '\n')
    print('Expected gradients (approx) - first five values only:\n')
    print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

    print('\nProgram paused. Press enter to continue.\n')
    raw_input()
    
    # ============= Part 2: Regularization and Accuracies =============
    # Optional Exercise:
    # In this part, you will get to try different values of lambda and
    # see how regularization affects the decision coundart
    # 
    # Try the following values of lambda (0, 1, 10, 100).
    # 
    # How does the decision boundary change when you vary lambda? How does
    # the training set accuracy vary?

    # Initialize fitting parameters
    initial_theta = np.zeros((np.shape(X)[1], 1))

    # Set regularization parameter lambda to 1 (you should vary this)
    lmbda = 1

    # Optimize
    xopt = optimize.minimize(fun=costFunctionReg, x0=np.transpose(initial_theta), args=(X, y, lmbda), method='TNC', jac=computeGradReg)
    theta = xopt.x

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)
    plt.title('lambda = ' + str(lmbda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.legend(('y = 1', 'y = 0', 'Decision boundary'))
    plt.show()

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: ' + str(np.mean(p.reshape((len(p), 1)) == y) * 100) + '\n')
    print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
    

if __name__ == '__main__':
    run()
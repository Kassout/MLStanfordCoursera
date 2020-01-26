import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize
from pip._vendor.distlib.compat import raw_input

from python.costFunction import costFunction, computeGrad
from python.plotData import plotData
from python.plotDecisionBoundary import plotDecisionBoundary

"""
    Machine Learning Online Class - Exercise 2: Logistic Regression

    Instructions
    ------------
 
    This file contains code that helps you get started on the logistic
    regression exercise. You will need to complete the following functions
    in this exercise:

    sigmoid.py
    costFunction.py
    predict.py
    costFunctionReg.py
    
    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
"""


def run():

    # ==================== Part 1: Plotting ====================
    # We start the exercise by first plotting the data to understand the
    # the problem we are working with.
    data = np.genfromtxt('ex2data1.csv', delimiter=',')
    X = data[:, [0, 1]]
    y = data[:, 2]

    print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'])

    plotData(X, y)

    # Put some labels and legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Specified in plot order
    plt.legend(('Admitted', 'Not admitted'))
    plt.show()

    print('\nProgram paused. Press enter to continue.\n')
    raw_input()
    
    # ============ Part 2: Compute Cost and Gradient ============
    # In this part of the exercise, you will implement the cost and gradient
    # for logistic regression. You neeed to complete the code in
    # costFunction.py

    # Setup the data matrix appropriately, and add ones for the intercept term
    m, n = np.shape(X)

    # Add intercept term to x and X_test
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros((n + 1, 1))

    # Compute and display initial cost and gradient
    cost, grad = (costFunction(initial_theta, X, y), computeGrad(initial_theta, X, y))

    print('Cost at initial theta (zeros): ' + str(cost) + '\n')
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): \n')
    print(grad, '\n')
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([[-24], [0.2], [0.2]])
    cost, grad = (costFunction(test_theta, X, y.to_numpy().reshape(len(y), 1)), computeGrad(test_theta, X, y.to_numpy().reshape(len(y), 1)))

    print('\nCost at test theta: ' + str(cost) + '\n')
    print('Expected cost (approx): 0.218\n')
    print('Gradient at test theta: \n')
    print(grad, '\n')
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

    print('\nProgram paused. Press enter to continue.\n')
    raw_input()
    
    # ============= Part 3: Optimizing using fminunc  =============
    # In this exercise, you will use a built-in function (fmin) to find the
    # optimal parameters theta.

    # Set options for fmin
    xopt = optimize.minimize(fun=costFunction, x0=initial_theta, args=(X, y), method='TNC', jac=computeGrad)

    print('Cost at theta found by scipy.optimize.minimize : ' + str(xopt.fun) + '\n')
    print('Expected cost (approx): 0.203\n')
    print('theta: \n')
    print(xopt.x, '\n')
    print('Expected theta (approx):\n')
    print(' -25.161\n 0.206\n 0.201\n')

    # Plot Boundary
    plotDecisionBoundary(xopt.x, X, y)

    #  Put some labels
    #  Labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Specified in plot order
    plt.legend(('Admitted', 'Not admitted'))

    print('\nProgram paused. Press enter to continue.\n')
    raw_input()


if __name__ == '__main__':
    run()
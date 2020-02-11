import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pip._vendor.distlib.compat import raw_input

from python.computeCost import computeCost
from python.gradientDescent import gradientDescent
from python.gradientDescentMulti import gradientDescentMulti
from python.normalEqn import normalEqn
from python.plotData import plotData
from python.featureNormalize import featureNormalize

"""
    Machine Learning Online Class
    Exercise 1: Linear regression with multiple variables

    Instructions
    ------------

    This file contains code that helps you get started on the
    linear regression exercise.

    You will need to complete the following functions in this
    exericse:

    plotData.py
    gradientDescent.py
    computeCost.py
    gradientDescentMulti.py
    computeCostMulti.py
    featureNormalize.py
    normalEqn.py

    For this part of the exercise, you will need to change some
    parts of the code below for various experiments (e.g., changing
    learning rates).
"""


def run():

    # ================ Part 1: Feature Normalization ================
    print('Loading data ...\n')

    # Load Data
    data = pd.read_csv('ex1data2.csv', sep=',', header=None)
    X = data.loc[:, 0:1]
    y = data.loc[:, 2]
    m = len(y)

    # Print out some data points
    print('First 10 examples from the dataset: \n')
    print('x = \n' + str(X.loc[0:9, :]) + '\n, y = \n' + str(y.loc[0:9]) + '\n')

    print('Program paused. Press enter to continue.\n')
    raw_input()

    # Scale features and set them to zero mean
    print('Normalizing Features ...\n')

    X, mu, sigma = featureNormalize(X)

    # Add intercept term to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # ================ Part 2: Gradient Descent ================
    print('Running gradient descent ...\n')

    # Choose some alpha value
    alpha = 1.2
    num_iters = 100

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    theta, J_history = gradientDescentMulti(X, y.to_numpy().reshape(len(y), 1), theta, alpha, num_iters)

    #  Plot the convergence graph
    plt.plot(list(range(np.size(J_history))), J_history, '-b', linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    #  Display gradient descent's result
    print('Theta computed from gradient descent: \n')
    print(str(theta) + '\n')
    print('\n')

    # Estimate the price of a 1650 sq-ft, 3 br house
    normalize_vec = np.array((np.array([1650, 3]) - mu) / sigma)
    price = np.concatenate((np.array([1]), normalize_vec))@theta

    print('Predicted price of a 1650 sq-ft, '
          '3 br house (using gradient descent):\n' + str(price) + '\n')

    print('Program paused. Press enter to continue.\n')
    raw_input()

    # ================ Part 3: Normal Equations ================
    print('Solving with normal equations...\n');

    # Load Data
    data = pd.read_csv('ex1data2.csv', sep=',', header=None)
    X = data.loc[:, 0:1]
    y = data.loc[:, 2]
    m = len(y)

    # Add intercept term to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # Display normal equation's result
    print('Theta computed from the normal equations: \n')
    print(str(theta) + '\n')
    print('\n')


    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.array([1, 1650, 3]) @ theta

    print('Predicted price of a 1650 sq-ft, '
          '3 br house (using gradient descent):\n' + str(price) + '\n')


if __name__ == '__main__':
    run()
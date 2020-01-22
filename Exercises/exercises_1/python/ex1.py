import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pip._vendor.distlib.compat import raw_input
from mpl_toolkits.mplot3d import Axes3D

from python.computeCost import computeCost
from python.gradientDescent import gradientDescent
from python.plotData import plotData

"""
Machine Learning Online Class - Exercise 1: Linear Regression

Instructions
------------

This file contains code that helps you get started on the
linear exercise. You will need to complete the following functions
in this exericse:

 plotData.m
 gradientDescent.m
 computeCost.m
 gradientDescentMulti.m
 computeCostMulti.m
 featureNormalize.m
 normalEqn.m

For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.

x refers to the population size in 10,000s
y refers to the profit in $10,000s
"""

def run():

    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...\n')
    data = pd.read_csv('ex1data1.csv', sep=',', header=None)
    X = data.loc[:, 0]
    y = data.loc[:, 1]
    # Number of training examples
    m = len(y)

    # Plot Data
    plotData(X, y)

    print('Program paused. Press enter to continue.\n')
    raw_input()

    # =================== Part 3: Cost and Gradient descent ===================

    # Add a column of ones to x
    X = np.concatenate((np.ones((m, 1)), data.loc[:, 0].to_numpy().reshape(len(X), 1)), axis=1)

    # initialize fitting parameters
    theta = np.zeros((2, 1))

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('\nTesting the cost function ...\n')
    # compute and display initial cost
    J = computeCost(X, y.to_numpy().reshape(len(y), 1), theta)
    print('With theta = [0  0]\nCost computed = ' + str(J) + '\n')
    print('Expected cost value (approx) 32.07\n')

    # further testing of the cost function
    J = computeCost(X, y.to_numpy().reshape(len(y), 1), np.array([-1, 2]))
    print('\nWith theta = [-1  2]\nCost computed = ' + str(J) + '\n')
    print('Expected cost value (approx) 54.24\n')

    print('Program paused. Press enter to continue.\n')
    raw_input()

    print('\nRunning Gradient Descent ...\n')
    # run gradient descX[:, 1]ent
    theta = gradientDescent(X, y.to_numpy().reshape(len(y), 1), theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:\n')
    print(str(theta[0]) + '\n' + str(theta[1]) + '\n')
    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')

    # Plot the linear fit
    fig, ax = plt.subplots()
    ax.plot(X[:, 1], y, 'rx', ms=10)
    ax.plot(X[:, 1], np.matmul(X, theta), '-')
    # Set the y axis label
    plt.ylabel('Profit in $10,000s')
    # Set the x axis label
    plt.xlabel('Population of City in 10,000s')
    plt.legend(labels=('Training data', 'Linear regression'))
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.matmul([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of ' + str(predict1*10000) + '\n')
    predict2 = np.matmul([1, 7], theta)
    print('For population = 70,000, we predict a profit of ' + str(predict2*10000) + '\n')

    print('Program paused. Press enter to continue.\n')
    raw_input()
    
    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...\n')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in range(0, len(theta0_vals)):
        for j in range(0, len(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
            J_vals[i, j] = computeCost(X, y.to_numpy().reshape(len(y), 1), t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = np.transpose(J_vals)

    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(X=theta0_vals, Y=theta1_vals, Z=J_vals, cmap=cm.coolwarm)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    plt.contour(theta0_vals, theta1_vals, J_vals, np.linspace(0.01, 100, 15).tolist())
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.plot(theta[0], theta[1], 'rx', ms=10)
    plt.show()


if __name__ == '__main__':
    run()


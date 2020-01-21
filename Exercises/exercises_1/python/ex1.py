import pandas as pd
import numpy as np
from pip._vendor.distlib.compat import raw_input

from python.computeCost import computeCost
from python.gradientDescent import gradientDescent
from python.plotData import plotData


def run():
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
    # run gradient descent
    theta = gradientDescent(X, y.to_numpy().reshape(len(y), 1), theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:\n')
    print(str(theta[0]) + '\n' + str(theta[1]) + '\n')
    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')

    # # Plot the linear fit
    # hold on # keep previous plot visible
    # plot(X(:,2), X*theta, '-')
    # legend('Training data', 'Linear regression')
    # hold off # don't overlay any more plots on this figure
    #
    # # Predict values for population sizes of 35,000 and 70,000
    # predict1 = [1, 3.5] *theta
    # print('For population = 35,000, we predict a profit of #f\n',...
    #     predict1*10000)
    # predict2 = [1, 7] * theta
    # print('For population = 70,000, we predict a profit of #f\n',...
    #     predict2*10000)
    #
    # print('Program paused. Press enter to continue.\n')
    # raw_input()


if __name__ == '__main__':
    run()

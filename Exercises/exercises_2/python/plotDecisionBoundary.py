from python.mapFeature import mapFeature
from python.plotData import plotData
import numpy as np
import matplotlib.pyplot as plt


def plotDecisionBoundary(theta, X, y):
    """
        plotDecisionBoundary() Plots the data points X and y into a new figure with
        the decision boundary defined by theta
        plotDecisionBoundary(theta, X,y) plots the data points with + for the
        positive examples and o for the negative examples. X is assumed to be
        a either
        1) Mx3 matrix, where the first column is an all-ones column for the
            intercept.
        2) MxN, N>3 matrix, where the first column is all-ones
    """

    # Plot Data
    plotData(X[:, 1:2], y)

    if len(X) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X.loc[:, 1])-2,  max(X.loc[:, 1])+2]

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1]) * (plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(('Admitted', 'Not admitted', 'Decision Boundary'))
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros(len(u), len(v))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i], v[j]) * theta

        # important to transpose z before calling contour
        z = np.transpose(z)

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
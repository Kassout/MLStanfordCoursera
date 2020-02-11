import matplotlib.pyplot as plt


def plotData(X, y):
    """
        plotData() Plots the data points X and y into a new figure
        plotData(x,y) plots the data points with + for the positive examples
        and o for the negative examples. X is assumed to be a Mx2 matrix.
    """

    # Find Indices of Positive and Negative Examples
    pos = [i for i, y in enumerate(y) if y == 1]
    neg = [i for i, y in enumerate(y) if y == 0]
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+', lw=2, ms=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=7)

import matplotlib.pyplot as plt


def plotData(x, y):
    # Plot the data
    plt.plot(x, y, 'rx', ms=10)
    # Set the y axis label
    plt.ylabel('Profit in $10,000s')
    # Set the x axis label
    plt.xlabel('Population of City in 10,000s')
    plt.show()

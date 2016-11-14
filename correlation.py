from matplotlib import pyplot
from pandas import read_csv
import numpy


def plot(data):
    correlations = data.corr()
    # plot correlation matrix
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    pyplot.show()

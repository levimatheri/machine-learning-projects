import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# PLOT DATA
def plot(dataset):
    scatter_matrix(dataset)
    plt.show()

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def plot(dataset):
    print("Plotting data")
    scatter_matrix(dataset)
    plt.show()

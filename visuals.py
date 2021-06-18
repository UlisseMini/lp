import numpy as np
import matplotlib.pyplot as plt


def plot(c, A, b):
    # TODO: more flexible abstraction

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # x >= 0 constraint
    # TODO: make this work
    # ax.plot_surface([-10, 10], [-10, 10], [0])
    # fig.show()


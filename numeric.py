from typing import Callable
import numpy as np
np.set_printoptions(suppress=True, precision=6)

def gradient(f: Callable, x: np.array, d = 1e-5):
    "Compute the gradient of f at x using central differences"
    g = np.zeros(len(x))

    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = d

        g[i] = (f(x + dx) - f(x - dx)) / (d*2)

    return g



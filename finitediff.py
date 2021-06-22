from typing import Callable
import numpy as np
np.set_printoptions(suppress=True, precision=6)

def grad(f: Callable, x: np.array, d = 1e-5):
    "Compute the gradient of f at x using central differences"
    g = np.zeros(len(x))

    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = d

        g[i] = (f(x + dx) - f(x - dx)) / (d*2)

    return g


if __name__ == '__main__':
    def f(x):
        return x[0]**2 + x[1]*2


    def nabla_f(x):
        return np.array([2*x[0], 2])


    xs = [
        [1,1]
    ]
    for x in xs:
        x = np.array(x)
        estimate = grad(f, x)
        real = nabla_f(x)
        if not np.allclose(estimate, real, atol=1e-3):
            print(f'want {real} got {estimate}')



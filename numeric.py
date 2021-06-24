from typing import Callable, List
import numpy as np
np.set_printoptions(suppress=True, precision=6)

def is_floating_point(x: np.ndarray) -> bool:
    "Predicate to avoid debugging hell when passed np.int64"
    return x.dtype in (np.float16, np.float32, np.float64)

def derivative(f: Callable, x: float, d = 1e-5):
    "Derivative of f at x"
    return (f(x + d) - f(x - d))/(d*2)


def derivative_wr(f: Callable, i: int, x: np.ndarray, d = 1e-5):
    "Derivative of f at x w/r to x[i]"
    assert is_floating_point(x)
    def _f(x_i):
        bak = x[i]; x[i] = x_i
        v = f(x)
        x[i] = bak
        return v

    return derivative(_f, x[i], d=d)

def derivative_wr_many(f: Callable, variables: List[int], x: np.ndarray, d=1e-5):
    "Derivative of f at x w/r to many variables (higher order)"
    assert is_floating_point(x)
    if len(variables) == 0: return f(x)
    if len(variables) == 1: return derivative_wr(f, variables[0], x, d=d)

    deriv = lambda x: derivative_wr(f, variables[0], x, d=d)
    return derivative_wr_many(deriv, variables[1:], x)


def gradient(f: Callable, x: np.ndarray, d = 1e-5):
    "Compute the gradient of f at x"
    assert is_floating_point(x)
    g = np.zeros(len(x))

    for i in range(len(x)):
        g[i] = derivative_wr(f, i, x, d=d)

    return g


def hessian(f: Callable, x: np.array, d = 1e-5):
    "Compute the hessian of f at x"
    assert is_floating_point(x)

    n = len(x)

    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h[i][j] = derivative_wr_many(f, [i, j], x, d=d)

    return h


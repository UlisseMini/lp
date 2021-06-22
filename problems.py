"Linear programming problems used for testing solvers"

import numpy as np

all_problems = []

def problem(c, A, b):
    p = np.array(c), np.array(A), np.array(b)
    all_problems.append(p)
    return p

R2 = problem(
    c=[-1, 2],
    A=[[1, 3],
       [2, 2]],
    b=[3, 2],
)

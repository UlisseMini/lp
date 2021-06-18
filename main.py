"""
Linear programming solver using interior point methods.

I use the simpler gradient step based methods instead of newton's method,
as this simplifies the code. Integrating newton's method will come later.

LP is
minimize c @ x
such that A @ x <= b and x >= 0
"""

import numpy as np
np.set_printoptions(suppress=True, precision=4)

# c = np.array([1, -2, 0.5])
# A = np.array([
#     [3, -1, 0],
#     [-3, 1, 4],
#     [0, 1, 1],
# ])
# b = np.array([1,2,4])

c = np.array([-1, 2])
A = np.array([
    [1, 3],
    [2, 2],
])
b = np.array([3,2])

def solve_ipm_newton(c, A, b):
    """
    Interior point works as follows
    1. Define a convex barrier function F(x) which
       approaches infinity as x approaches a constraint.
    2. Define f_t(x) = t*c@x + F(x) where t is tempature,
       how much we optimize cost vs barrier.
    3. Obtain x_0 as the optimal point where t=0. the center
       of the barrier hyperplanes.
    4. Increase t "as much as possible" such that newton's method
       still converges.
    5. Repeat (4) until a solution is found to required accuracy.
    """
    raise NotImplementedError()


def solve_ipm_gradient(c, A, b):
    """
    This is a simplified interior point method.
    1. Define a convex barrier function F(x) which
       approaches infinity as x approaches as constraint.
    2. Find x_0 as the minimum of F(x), the center of the
       barrier hyperplanes.
    3. Minimize f(x) = c@x + F(x) using gradient decent.
    """

    # 1. Define the barrier function
    def F(x):
        # use logs to make expr explode as x->0
        # Ax <= b <=> b - Ax >= 0
        return -np.sum(np.log(b - A@x)) + np.sum(np.log(x))

    # 1.5: Compute the gradient of the barrier function
    def nabla_F(x):
        # gradient of -np.sum(np.log(b - A@x))
        # TODO: Make sure this is right, learn vector calculus so I'm not a sad boi
        first = x * sum((A[:,j]) / (b - A@x) for j in range(len(b)))
        # gradient of np.sum(log(x))
        second = (1/x)
        return first + second

    # import IPython; IPython.embed()
    # 2. Find x_0 as the minimum of the barrier, since the
    #    barrier is convex this is where the gradient is zero.

    # 3. Minimize f(x) = c@x + F(x) using gradient decent

    return np.array([0, 0])


def solve_scipy(c, A, b):
    import scipy.optimize
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b)
    assert res.success

    return res.x


def test_solver(solver):
    x = solver(c, A, b)

    if not (A@x <= b).all() or not (x >= 0).all():
        print("!! INVALID SOLUTION")
        print(f"A @ {x} <= b = {A@x <= b}")

    print(f'cost {c @ x:.4f} x: {x}')


def test():
    test_solver(solve_scipy)
    test_solver(solve_ipm_gradient)


def plot():
    import visuals
    visuals.plot(c, A, b)


def main():
    import sys
    if len(sys.argv) < 2:
        print(f'Usage: python main.py <test/plot>')
        exit()

    if sys.argv[1] == 'test': test()
    if sys.argv[1] == 'plot': plot()

if __name__ == '__main__':
    main()

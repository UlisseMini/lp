import numpy as np
from functools import partial
import numeric

def newton_step(x, gradient, hessian):
    """
    Preform a newton's method step returning x_{n+1} given x_n
    Details: https://uli.rocks/p/interior-point
    """
    # TODO: compute the inverse hessian symbolically (faster)
    return x - np.linalg.inv(hessian(x)) @ gradient(x)


def optimize(step_fn, x, atol=1e-3, max_iters=100):
    i = 0
    while i < max_iters:
        new_x = step_fn(x)
        if np.allclose(new_x, x, atol=atol):
            return new_x
        x = new_x
        i += 1

    raise ValueError(f"did not converge after {max_iters} iterations x={x}")


def barrier(x, A, b):
    # use logs to make expr explode as x->0
    # Ax <= b <=> b - Ax >= 0
    return -np.sum(np.log(b - A@x)) - np.sum(np.log(x))


def barrier_gradient(x, A, b):
    # Derivation (paste into https://latexeditor.lagrida.com/)
    # \nabla\left(\log(b_1-a_1^Tx) + \dots + \log(b_n - a_n^Tx)\right)
    # = \frac{-a_1}{b_1 - a_1^Tx} + \dots + \frac{-a_n}{b_n - a_n^Tx}
    #
    # The -(1/x) part is straightforward by linearity

    return -sum((-A[i]/(b[i] - A[i].T @ x) for i in range(len(b)))) - (1/x)


def barrier_hessian(x, A, b):
    # TODO: explicit hessian formula (or autodiff)
    return numeric.hessian(lambda x: barrier(x, A, b), x)


def solve_ipm_newton(c, A, b, atol=1e-3):
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
    See https://youtu.be/4mpq-wsYBxw and my notes https://uli.rocks/p/interior-point
    """

    # 2. TODO Find x_0 as the minimum of the barrier, since the
    #    barrier is convex this is where the gradient is zero.
    # F = partial(barrier, A=A, b=b) # not used
    nabla_F = partial(barrier_gradient, A=A, b=b)

    x = np.array([0.3, 0.3])

    # 3. Minimize f(x) = c*c@x + F(x) using gradient decent
    t = 1
    def gradient_f(x):
        return t*c + nabla_F(x)

    def hessian_f(x):
        return barrier_hessian(x, A=A, b=b)

    step_fn = partial(newton_step, gradient=gradient_f, hessian=hessian_f)
    while True:
        new_x = optimize(step_fn, x)
        if np.allclose(new_x, x, atol=atol):
            break

        x = new_x
        # TODO: Find nu for log barrier
        t *= 5/4


    return x

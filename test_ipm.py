import unittest
import ipm
import numeric
import numpy as np
from functools import partial
import problems


_, A, b = problems.R2

barrier = partial(ipm.barrier, A=A, b=b)
barrier_gradient = partial(ipm.barrier_gradient, A=A, b=b)
barrier_hessian = partial(ipm.barrier_hessian, A=A, b=b)
xs = [
    np.array(x, dtype=np.float64) for x in
    [[0.3, 0.3]]
]

class TestBarrierGradient(unittest.TestCase):
    def test_barrier_gradient(self):
        for x in xs:
            want = numeric.gradient(barrier, x)
            got = barrier_gradient(x)
            self.assertTrue(
                np.allclose(want, got),
                f"\nwant gradient\n{want}\ngot gradient\n{got}"
            )


    def test_barrier_hessian(self):
        for x in xs:
            want = numeric.hessian(barrier, x)
            got = barrier_hessian(x)

            self.assertTrue(
                np.allclose(want, got),
                f"\nwant hessian\n{want}\ngot hessian\n{got}"
            )


import unittest
import ipm
import finitediff
import numpy as np
from functools import partial

class TestBarrierGradient(unittest.TestCase):
    def test_barrier_gradient(self):
        A = np.array([
            [1, 3],
            [2, 2],
        ])
        b = np.array([3,2])

        barrier = partial(ipm.barrier, A=A, b=b)
        barrier_gradient = partial(ipm.barrier_gradient, A=A, b=b)

        xs = [[0.3, 0.3]]
        for x in xs:
            x = np.array(x)
            want = finitediff.grad(barrier, x)
            got = barrier_gradient(x)
            self.assertTrue(
                np.allclose(want, got, atol=1e-5),
                f"want gradient {want} got gradient {got}"
            )





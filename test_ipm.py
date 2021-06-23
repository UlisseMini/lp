import unittest
import ipm
import numeric
import numpy as np
from functools import partial
import problems

class TestBarrierGradient(unittest.TestCase):
    def test_barrier_gradient(self):
        _, A, b = problems.R2

        barrier = partial(ipm.barrier, A=A, b=b)
        barrier_gradient = partial(ipm.barrier_gradient, A=A, b=b)

        xs = [[0.3, 0.3]]
        for x in xs:
            x = np.array(x)
            want = numeric.gradient(barrier, x)
            got = barrier_gradient(x)
            self.assertTrue(
                np.allclose(want, got, atol=1e-5),
                f"want gradient {want} got gradient {got}"
            )





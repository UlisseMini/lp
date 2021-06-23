import unittest
import numpy as np
import numeric

def f(x):
    return x[0]**2 + x[1]*2


def nabla_f(x):
    return np.array([2*x[0], 2])


class NumericalTest(unittest.TestCase):
    def testGradent(self):
        xs = [
            [1,1]
        ]
        for x in xs:
            x = np.array(x)
            estimate = numeric.gradient(f, x)
            real = nabla_f(x)
            self.assertTrue(
                np.allclose(estimate, real, atol=1e-3),
                f'want {real} got {estimate}'
            )



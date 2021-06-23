import unittest
import numpy as np
import numeric

def f(x):
    return x[0]**2 + 2*x[1]

def nabla_f(x):
    return np.array([2*x[0], 2])


class TestNumerics(unittest.TestCase):
    def test_gradient(self):
        xs = [
            [1,1], [3,1], [5,5], [-1,3],
        ]
        for x in xs:
            x = np.array(x)
            estimate = numeric.gradient(f, x)
            real = nabla_f(x)
            print('real', real)
            print('estimate', estimate)
            self.assertTrue(
                np.allclose(estimate, real, atol=1e-3),
                f'want {real} got {estimate}'
            )



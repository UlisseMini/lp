import unittest
import numpy as np
import numeric

def f(v):
    x, y = v
    return y * x**2 + 2*y**3 * x

def gradient_f(v):
    x, y = v
    return np.array([y*2*x + 2*y**3, x**2 + 6*y**2 * x])

def hessian_f(v):
    x, y = v
    return np.array([
        [2*y, 2*x + 6*y**2],
        [2*x + 6*y**2, 12*x*y],
    ])

class TestNumerics(unittest.TestCase):
    def test_numerics(self):
        xs = [
            [1,1], [3,1], [5,5], [-1,3],
        ]
        atol = 1e-3
        for x in xs:
            # fp32 doesn't work well, i don't know numerical analysis send help pls
            x = np.array(x).astype(np.float64)
            with self.subTest(f"x = {x}"):
                estimate = numeric.gradient(f, x)
                real = gradient_f(x)
                self.assertTrue(
                    np.allclose(estimate, real, atol=atol),
                    f'want {real} got {estimate}'
                )

                estimate_h = numeric.hessian(f, x)
                real_h = hessian_f(x)
                self.assertTrue(
                    np.allclose(estimate_h, real_h, atol=atol),
                    f'want {real_h} got {estimate_h}'
                )

    def test_derivative(self):
        def _f(x):
            return 3*x**2 + 2

        def _deriv(x):
            return 6*x

        xs = [1,2,3]
        for x in xs:
            x = np.array(x).astype(np.float32)
            self.assertAlmostEqual(
                _deriv(x),
                numeric.derivative(_f, x)
            )

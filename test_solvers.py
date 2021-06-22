import unittest
import solvers
import numpy as np

precision = 5

reference_solver = solvers.solve_scipy

class TestSolvers(unittest.TestCase):
    def solver_test(self, solver):
        c = np.array([-1, 2])
        A = np.array(
            [
                [1, 3],
                [2, 2],
            ]
        )
        b = np.array([3, 2])

        x = solver(c, A, b).round(precision)

        with self.subTest("x obeys constraints"):
            x_ge_zero = (x >= 0).all()
            self.assertTrue(x_ge_zero, f"{x} is less then zero")

            ax_le_b = (A @ x) <= b
            self.assertTrue(ax_le_b.all(), f"Ax <= b is {ax_le_b}")

        with self.subTest("x is optimal"):
            x_star = reference_solver(c, A, b).round(precision)
            self.assertTrue((x == x_star).all(), f"got {x} want {x_star}")

    def test_scipy(self):
        # if this fails, you done goofed
        self.solver_test(reference_solver)

    def test_ipm_gradient(self):
        self.solver_test(solvers.solve_ipm_gradient)

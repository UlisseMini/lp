import unittest
import solvers
import problems
import numpy as np

reference_solver = solvers.solve_scipy

precision = 2
atol = 10 ** -precision

class TestSolvers(unittest.TestCase):
    def solver_test(self, solver):
        for problem in problems.all_problems:
            # TODO: Add description to each problem to ease debugging
            c, A, b = problem

            x = solver(c, A, b)

            with self.subTest("x obeys constraints"):
                x_ge_zero = (x >= 0).all()
                self.assertTrue(x_ge_zero, f"{x} is less then zero")

                ax_le_b = (A @ x).round(precision) <= b
                self.assertTrue(ax_le_b.all(), f"Ax <= b is {ax_le_b}")

            with self.subTest("x is optimal"):
                x_star = reference_solver(c, A, b)
                self.assertTrue(np.allclose(x, x_star, atol=atol), f"got {x} want {x_star}")

    def test_scipy(self):
        # if this fails, you done goofed
        self.solver_test(reference_solver)

    def test_ipm_newton(self):
        self.solver_test(solvers.solve_ipm_newton)

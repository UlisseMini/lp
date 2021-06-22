from ipm import solve_ipm_gradient, solve_ipm_newton

def solve_scipy(c, A, b):
    import scipy.optimize
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b)
    assert res.success

    return res.x



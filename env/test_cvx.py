import cvxpy as cp

A = cp.Variable()
B = cp.Variable()
C = cp.Variable()
constraints = [(A) @ C >= 0]
problem = cp.Problem(cp.Minimize(A + B - C), constraints)
problem.solve()

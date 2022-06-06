import cvxpy as cp

# Create two scalar optimization variables.

n_hvac = 10
rolling_horizon = 60

HVAC_state = cp.Variable((n_hvac,rolling_horizon))
constraints = [HVAC_state ==0]
problem = cp.Problem(cp.Minimize(cp.sum_squares(HVAC_state)), constraints)
problem.solve()

print(HVAC_state.value)
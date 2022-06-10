import time
import numpy as np
import cvxpy as cp


rolling_horizon = 1
nb_agent = 202  # 201 make it fail -- guro

HVAC_state = cp.Variable((rolling_horizon, nb_agent))
constraints = [HVAC_state == 1]
problem = cp.Problem(cp.Minimize(cp.sum_squares(HVAC_state)), constraints)

print(cp.installed_solvers())
start = time.time()
problem.solve(solver=cp.GUROBI, NodefileStart=0.1)

end = time.time()
print(problem.status)
print(HVAC_state.value)

print(end - start)

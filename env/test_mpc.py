from re import I
from tabnanny import verbose
import time
import numpy as np
import cvxpy as cp


hvac_power = np.array([1]*60)
initial_temperature = np.array([24]*60)
target_temperature = np.array([20]*60)


rolling_horizon = 10
signal = np.array([1]*rolling_horizon)
nb_agent = 5

HVAC_state =  cp.Variable((rolling_horizon, nb_agent), boolean = True)
temperature = cp.Variable((rolling_horizon, nb_agent))
consumption = cp.Variable(rolling_horizon)


temperature_difference = cp.Variable((rolling_horizon, nb_agent))

constraints = [consumption[i] == cp.sum(cp.multiply(HVAC_state[i], hvac_power[:nb_agent]))  for i in range(rolling_horizon)]
constraints += [temperature[0] == initial_temperature[:nb_agent]]
constraints += [temperature_difference[i] == (temperature[i] - target_temperature[:nb_agent])  for i in range(1,rolling_horizon)]

print(np.size(temperature_difference))
constraints += [temperature[i][j] == temperature[i-1][j] -  hvac_power[j] * HVAC_state[i][j] for i in range(1,rolling_horizon) for j in range(nb_agent)]
problem = cp.Problem(cp.Minimize(cp.sum_squares(consumption-signal) + cp.sum_squares(temperature_difference)/np.size(temperature_difference) ) , constraints)

print(cp.installed_solvers())
start = time.time()
problem.solve(solver=cp.GUROBI, NodefileStart  = 0.1)

end = time.time()
print(problem.status)
print(HVAC_state.value)
print(temperature.value)

print(end - start)

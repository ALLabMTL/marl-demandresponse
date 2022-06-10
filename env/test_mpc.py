from re import I
from tabnanny import verbose
import time
import numpy as np
import cvxpy as cp


hvac_power = np.array([1] * 60)
initial_temperature = np.array([24] * 60)
target_temperature = np.array([20] * 60)
remaining_lockout = [9, 9, 5, 5, 5, 5, 5, 5, 5, 5]


rolling_horizon = 15
nb_agent = 10
lockout_duration = 10
signal = np.array([5] * rolling_horizon)

past = np.zeros((nb_agent, lockout_duration))

for i, remaining_t in enumerate(remaining_lockout):
    past[i][:remaining_t] = 1
past = past.transpose()


HVAC_state = cp.Variable((rolling_horizon + lockout_duration, nb_agent), boolean=True)


temperature = cp.Variable((rolling_horizon, nb_agent))
consumption = cp.Variable(rolling_horizon)


temperature_difference = cp.Variable((rolling_horizon, nb_agent))


constraints = [
    consumption[i]
    == cp.sum(cp.multiply(HVAC_state[i + lockout_duration], hvac_power[:nb_agent]))
    for i in range(rolling_horizon)
]
constraints += [HVAC_state[:lockout_duration] == past]
constraints += [temperature[0] == initial_temperature[:nb_agent]]
constraints += [
    lockout_duration
    * (HVAC_state[i + lockout_duration, k] - HVAC_state[i + lockout_duration - 1, k])
    - (
        cp.sum(
            [
                1 - HVAC_state[i + lockout_duration - j - 1, k]
                for j in range(lockout_duration)
            ]
        )
    )
    <= 0
    for i in range(rolling_horizon)
    for k in range(nb_agent)
]
constraints += [
    temperature_difference[i] == (temperature[i] - target_temperature[:nb_agent])
    for i in range(1, rolling_horizon)
]

print(np.size(temperature_difference))
constraints += [
    temperature[i][j]
    == temperature[i - 1][j] - hvac_power[j] * HVAC_state[i + lockout_duration][j]
    for i in range(1, rolling_horizon)
    for j in range(nb_agent)
]
problem = cp.Problem(
    cp.Minimize(
        cp.sum_squares(consumption - signal)
        + cp.sum_squares(temperature_difference) / np.size(temperature_difference)
    ),
    constraints,
)

print(cp.installed_solvers())
start = time.time()
problem.solve(solver=cp.GUROBI, NodefileStart=0.5)

end = time.time()
print(problem.status)
print(HVAC_state.value)
# print(temperature.value)

print(end - start)


print(
    lockout_duration
    * (
        HVAC_state.value[0 + lockout_duration, 2]
        - HVAC_state.value[0 + lockout_duration - 1, 2]
    )
)
print(
    cp.sum(
        [
            1 - HVAC_state.value[8 + lockout_duration - j, 2]
            for j in range(lockout_duration)
        ]
    )
)

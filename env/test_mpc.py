from re import I
from tabnanny import verbose
import time
import numpy as np
import cvxpy as cp


hvac_power = np.array([1] * 15)
initial_air_temperature = np.array([24] * 15)
initial_mass_temperature = np.array([24] * 15)
target_temperature = np.array([20] * 15)
remaining_lockout = [9, 9, 5, 5, 5, 5, 5, 5, 5, 5]
temperature_extérieur = 20

rolling_horizon = 15
nb_agent = 10
lockout_duration = 10
signal = np.array([5] * rolling_horizon)


# ____________

time_step_sec = 4
Hm, Ca, Ua, Cm = (
    np.array([1] * 15),
    np.array([1] * 15),
    np.array([1] * 15),
    np.array([1] * 15),
)

# Convert Celsius temperatures in Kelvin
od_temp_K = 30 + 273


# Total heat addition to air
other_Qa = np.array([1] * 15)  # Solar gain
Qa = cp.Variable((rolling_horizon, nb_agent))
# Heat from inside devices (oven, windows, etc)


# Variables and time constants
a = (Cm[i] * Ca[i] / Hm[i] for i in range(15))
b = (Cm[i] * (Ua[i] + Hm[i]) / Hm[i] + Ca[i] for i in range(15))
c = (Ua[i] for i in range(15))
d = (Qa[i] + Ua[i] * od_temp_K[i] for i in range(15))


r1 = ((-b[i] + np.sqrt(b[i] ** 2 - 4 * a[i] * c[i])) / (2 * a[i]) for i in range(15))
r2 = ((-b[i] - np.sqrt(b[i] ** 2 - 4 * a[i] * c[i])) / (2 * a[i]) for i in range(15))

dTA0dt = cp.Variable((rolling_horizon, nb_agent))
A1 = cp.Variable((rolling_horizon, nb_agent))
A2 = cp.Variable((rolling_horizon, nb_agent))
A3 = (r1[i] * Ca[i] / Hm[i] + (Ua[i] + Hm[i]) / Hm[i] for i in range(15))
A4 = (r2[i] * Ca[i] / Hm[i] + (Ua[i] + Hm[i]) / Hm[i] for i in range(15))

# Updating the temperature

new_current_temp_K = cp.Variable((rolling_horizon, nb_agent))
new_current_mass_temp_K = cp.Variable((rolling_horizon, nb_agent))

past = np.zeros((nb_agent, lockout_duration))

for i, remaining_t in enumerate(remaining_lockout):
    past[i][:remaining_t] = 1
past = past.transpose()


HVAC_state = cp.Variable((rolling_horizon + lockout_duration, nb_agent), boolean=True)
air_temperature = cp.Variable((rolling_horizon, nb_agent))
mass_temperature = cp.Variable((rolling_horizon, nb_agent))
consumption = cp.Variable(rolling_horizon)


temperature_difference = cp.Variable((rolling_horizon, nb_agent))


constraints = [
    consumption[i]
    == cp.sum(cp.multiply(HVAC_state[i + lockout_duration], hvac_power[:nb_agent]))
    for i in range(rolling_horizon)
]
constraints += [
    Qa[t][i] == HVAC_state[t][i] * hvac_power[t][i] + other_Qa[t]
    for t in range(rolling_horizon)
    for i in range(nb_agent)
]
constraints += [HVAC_state[:lockout_duration] == past]
constraints += [air_temperature[0] == initial_air_temperature[:nb_agent]]
constraints += [air_temperature[0] == initial_mass_temperature[:nb_agent]]
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
    temperature_difference[i] == (air_temperature[i] - target_temperature[:nb_agent])
    for i in range(1, rolling_horizon)
]

print(np.size(temperature_difference))
constraints += [
    air_temperature[i][j]
    == air_temperature[i - 1][j] - hvac_power[j] * HVAC_state[i + lockout_duration][j]
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

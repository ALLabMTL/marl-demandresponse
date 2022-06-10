from re import I
from tabnanny import verbose
import time
import numpy as np
import cvxpy as cp


hvac_power = np.array([1] * 60)
initial_air_temperature = np.array([24] * 60)
initial_mass_temperature = np.array([24] * 60)
target_temperature = np.array([20] * 60)
remaining_lockout = [9, 9, 5, 5, 5, 5, 5, 5, 5, 5]
temperature_extérieur = 20

rolling_horizon = 15
nb_agent = 10
lockout_duration = 10
signal = np.array([5] * rolling_horizon)


# ____________

time_step_sec = 4
Hm, Ca, Ua, Cm = (
    np.array([1] * 60),
    np.array([1] * 60),
    np.array([1] * 60),
    np.array([1] * 60),
)

# Convert Celsius temperatures in Kelvin
od_temp_K = od_temp + 273
current_temp_K = self.current_temp + 273
current_mass_temp_K = self.current_mass_temp + 273

# Heat from hvacs (negative if it is AC)
total_Qhvac = 0
for hvac_id in self.hvacs_ids:
    hvac = self.hvacs[hvac_id]
    total_Qhvac += hvac.get_Q()

# Total heat addition to air
other_Qa = self.house_solar_gain(date_time)  # windows, ...
Qa = total_Qhvac + other_Qa
# Heat from inside devices (oven, windows, etc)
Qm = 0

# Variables and time constants
a = Cm * Ca / Hm
b = Cm * (Ua + Hm) / Hm + Ca
c = Ua
d = Qm + Qa + Ua * od_temp_K
g = Qm / Hm

r1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
r2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

dTA0dt = (
    Hm * current_mass_temp_K / Ca
    - (Ua + Hm) * current_temp_K / Ca
    + Ua * od_temp_K / Ca
    + Qa / Ca
)

A1 = (r2 * current_temp_K - dTA0dt - r2 * d / c) / (r2 - r1)
A2 = current_temp_K - d / c - A1
A3 = r1 * Ca / Hm + (Ua + Hm) / Hm
A4 = r2 * Ca / Hm + (Ua + Hm) / Hm

# Updating the temperature
old_temp_K = current_temp_K
new_current_temp_K = (
    A1 * np.exp(r1 * time_step_sec) + A2 * np.exp(r2 * time_step_sec) + d / c
)
new_current_mass_temp_K = (
    A1 * A3 * np.exp(r1 * time_step_sec)
    + A2 * A4 * np.exp(r2 * time_step_sec)
    + g
    + d / c
)

self.current_temp = new_current_temp_K - 273
self.current_mass_temp = new_current_mass_temp_K - 273
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

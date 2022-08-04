import numpy as np
import cvxpy as cp
import time
import sys
import os

id_rng = np.random.default_rng()
unique_ID = str(int(id_rng.random() * 1000000))

def best_MPC_action(
    nb_agent,
    hvac_cooling,
    hvac_consumption,
    initial_air_temperature,
    initial_mass_temperature,
    target_temperature,
    remaining_lockout,
    signal,
    od_temp_K,
    other_Qa,
    Hm,
    Ca,
    Ua,
    Cm,
    rolling_horizon,
    time_step_sec,
    lockout_duration,
):

    path = "./log_gurobi"+ unique_ID +".txt"
    sys.stdout = open(path, "w")

    Qa = cp.Variable((rolling_horizon, nb_agent))

    a = [Cm[i] * Ca[i] / Hm[i] for i in range(nb_agent)]
    b = [Cm[i] * (Ua[i] + Hm[i]) / Hm[i] + Ca[i] for i in range(nb_agent)]
    c = [Ua[i] for i in range(nb_agent)]
    d = cp.Variable((rolling_horizon, nb_agent))

    r1 = [
        (-b[i] + np.sqrt(b[i] ** 2 - 4 * a[i] * c[i])) / (2 * a[i])
        for i in range(nb_agent)
    ]
    r2 = [
        (-b[i] - np.sqrt(b[i] ** 2 - 4 * a[i] * c[i])) / (2 * a[i])
        for i in range(nb_agent)
    ]

    dTA0dt = cp.Variable((rolling_horizon, nb_agent))
    A1 = cp.Variable((rolling_horizon, nb_agent))
    A2 = cp.Variable((rolling_horizon, nb_agent))
    A3 = [r1[i] * Ca[i] / Hm[i] + (Ua[i] + Hm[i]) / Hm[i] for i in range(nb_agent)]
    A4 = [r2[i] * Ca[i] / Hm[i] + (Ua[i] + Hm[i]) / Hm[i] for i in range(nb_agent)]

    past = np.zeros((nb_agent, lockout_duration))

    for i, remaining_t in enumerate(remaining_lockout):
        past[i][:remaining_t] = 1
    past = past.transpose()

    HVAC_state = cp.Variable(
        (rolling_horizon + lockout_duration, nb_agent), boolean=True
    )
    air_temperature = cp.Variable((rolling_horizon, nb_agent))
    mass_temperature = cp.Variable((rolling_horizon, nb_agent))
    consumption = cp.Variable(rolling_horizon)

    temperature_difference = cp.Variable((rolling_horizon, nb_agent))

    constraints = [
        consumption[t]
        == cp.sum(
            cp.multiply(HVAC_state[t + lockout_duration], hvac_consumption[:nb_agent])
        )
        for t in range(rolling_horizon)
    ]

    constraints += [
        d[t][i] == Qa[t][i] + Ua[i] * od_temp_K[t]
        for t in range(rolling_horizon)
        for i in range(nb_agent)
    ]

    constraints += [
        Qa[t][i] == -HVAC_state[t + lockout_duration][i] * hvac_cooling[i] + other_Qa[t]
        for t in range(rolling_horizon)
        for i in range(nb_agent)
    ]

    constraints += [
        dTA0dt[t][i]
        == Hm[i] * mass_temperature[t][i] / Ca[i]
        - (Ua[i] + Hm[i]) * air_temperature[t][i] / Ca[i]
        + Ua[i] * od_temp_K[t] / Ca[i]
        + Qa[t][i] / Ca[i]
        for t in range(rolling_horizon)
        for i in range(nb_agent)
    ]

    constraints += [
        A1[t][i]
        == (r2[i] * air_temperature[t][i] - dTA0dt[t][i] - r2[i] * d[t][i] / c[i])
        / (r2[i] - r1[i])
        for t in range(rolling_horizon)
        for i in range(nb_agent)
    ]

    constraints += [
        A2[t][i] == air_temperature[t][i] - d[t][i] / c[i] - A1[t][i]
        for t in range(rolling_horizon)
        for i in range(nb_agent)
    ]

    constraints += [
        air_temperature[t + 1][i]
        == A1[t][i] * np.exp(r1[i] * time_step_sec)
        + A2[t][i] * np.exp(r2[i] * time_step_sec)
        + d[t][i] / c[i]
        for t in range(0, rolling_horizon - 1)
        for i in range(nb_agent)
    ]

    constraints += [
        mass_temperature[t + 1][i]
        == A1[t][i] * A3[i] * np.exp(r1[i] * time_step_sec)
        + A2[t][i] * A4[i] * np.exp(r2[i] * time_step_sec)
        + d[t][i] / c[i]
        for t in range(0, rolling_horizon - 1)
        for i in range(nb_agent)
    ]

    constraints += [HVAC_state[:lockout_duration] == past]
    constraints += [air_temperature[0] == initial_air_temperature[:nb_agent]]
    constraints += [mass_temperature[0] == initial_mass_temperature[:nb_agent]]
    constraints += [
        lockout_duration
        * (
            HVAC_state[i + lockout_duration, k]
            - HVAC_state[i + lockout_duration - 1, k]
        )
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
        temperature_difference[t][i] == (air_temperature[t][i] - target_temperature[i])
        for t in range(i, rolling_horizon)
        for i in range(nb_agent)
    ]

    problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(consumption - signal) / (7500**2) *0
            + cp.sum_squares(temperature_difference)
        ),
        constraints,
    )

    problem.solve(solver=cp.GUROBI, NodefileStart=0.5)

    os.remove(path)
    sys.stdout = sys.__stdout__
    print(temperature_difference.value)
    print(HVAC_state.value[lockout_duration:])
    return (HVAC_state.value > 0.5)[lockout_duration, :]


# hvac_power = np.array([10000] * 15)15
# initial_air_temperature = np.array([16 + 273] * 15)
# initial_mass_temperature = np.array([16 + 273] * 15)
# target_temperature = np.array([20 + 273] * 15)
# remaining_lockout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# rolling_horizon = 15
# nb_agent = 10
# lockout_duration = 10
# signal = np.array([50000] * rolling_horizon)
# od_temp_K = np.array([30 + 273] * 15)
# other_Qa = np.array([1] * 15)  # Solar gain

# time_step_sec = 4
# Hm, Ca, Ua, Cm = (
#     np.array([2.84e03] * 15),
#     np.array([9.08e05] * 15),
#     np.array([2.18e02] * 15),
#     np.array([3.45e06] * 15),
# )
# # ____________

# start = time.time()

# next_action = best_MPC_action(
#     nb_agent,
#     hvac_power,
#     initial_air_temperature,
#     initial_mass_temperature,
#     target_temperature,
#     remaining_lockout,
#     signal,
#     od_temp_K,
#     other_Qa,
#     Hm,
#     Ca,
#     Ua,
#     Cm,
#     rolling_horizon,
#     time_step_sec,
#     lockout_duration,
# )


# end = time.time()
# print(next_action)
# print(end - start)

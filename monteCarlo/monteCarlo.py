import sys

sys.path.insert(1, "../marl-demandresponse")
sys.path.append("..")

import argparse
import copy
import datetime
import itertools as it
import math
import time
from datetime import date, timedelta

import pandas as pd
import json
import csv
from agents import *
from config import config_dict
from env import *
from utils import get_actions

SECOND_IN_A_HOUR = 3600
NB_TIME_STEPS_BY_SIM = 75
NB_TIME_STEPS_AVG = 10

parser = argparse.ArgumentParser(description="Deployment options")

parser.add_argument(
    "--lower_fraction",
    type=float,
    default=0,
    help="Lower fraction of the montecarlo simulation while run concurently",
)

parser.add_argument(
    "--upper_fraction",
    type=float,
    default=1,
    help="Upper fraction of the montecarlo simulation while run concurently",
)

opt = parser.parse_args()
""" To test
parameters_dict = {
    "Ua_ratio": [1],#[0.9, 1, 1.1],
    "Cm_ratio": [1],#[0.9, 1, 1.1],
    "Ca_ratio": [1],#[0.9, 1, 1.1],
    "Hm_ratio": [1],#[0.9, 1, 1.1],
    "air_temp": [1],  # Setter au debut. Si c'est plus haut, ne fait pas de différence sur 75 time steps.
    "mass_temp": [0],  # Setter au debut, ajouter au conf dict
    "OD_temp": [15],  # fixé
    "HVAC_power": [15000],
    "hour": [
        0.0,
        #3.0,
        #6.0,
        #7.0,
        #7.50,
        #11.0,
        #13.0,
        #16.0,
        #17.0,
        #17.5,
        #21.0,
        #24 - 1.0 / 3600,
    ],
    "date": [
        (2021, 1, 1),
        #(2021, 3, 21),
        #(2021, 6, 21),
        #(2021, 9, 21),
        #(2021, 12, 21),
        #(2021, 12, 31),
    ],
}
"""
parameters_dict = {
    "Ua_ratio": [0.9, 1, 1.1],
    "Cm_ratio": [0.9, 1, 1.1],
    "Ca_ratio": [0.9, 1, 1.1],
    "Hm_ratio": [0.9, 1, 1.1],
    "air_temp": [-4, -2, -1, -0.3, 0, 0.3, 1, 2, 4],  # Setter au debut. Si c'est plus haut, ne fait pas de différence sur 75 time steps.
    "mass_temp": [-4, -2, 0, 2, 4],  # Setter au debut, ajouter au conf dict
    "OD_temp": [1, 3, 5, 7, 9, 11, 13, 15],  # fixé
    "HVAC_power": [10000, 15000],
    "hour": [
        0.0,
        3.0,
        6.0,
        7.0,
        7.50,
        11.0,
        13.0,
        16.0,
        17.0,
        17.5,
        21.0,
        24 - 1.0 / 3600,
    ],
    "date": [
        (2021, 1, 1),
        (2021, 3, 21),
        (2021, 6, 21),
        (2021, 9, 21),
        (2021, 12, 21),
        (2021, 12, 31),
    ],
}

d0 = date(2021, 1, 1)
parameters_dict["date"] = [
    (date(x[0], x[1], x[2]) - d0).days for x in parameters_dict["date"]
]
parameters_dict["hour"] = [x * SECOND_IN_A_HOUR for x in parameters_dict["hour"]]

with open('interp_parameters_dict.json', 'w') as fp:
    json.dump(parameters_dict, fp)
with open('interp_dict_keys.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["Ua_ratio", "Cm_ratio", "Ca_ratio", "Hm_ratio", "air_temp", "mass_temp", "OD_temp", "HVAC_power", "hour", "date"])

keys = parameters_dict.keys()
combinations = it.product(*(parameters_dict[Name] for Name in keys))


def number_of_combination(dict):
    nb_comb = 1
    for key in dict:
        nb_comb *= len(dict[key])
    return nb_comb


def eval_parameters_bangbang_average_consumption(
    Ua_ratio, Cm_ratio, Ca_ratio, Hm_ratio, air_temp, mass_temp, OD_temp, HVAC_power, hour, date
):

    config = copy.deepcopy(config_dict)
    date = d0 + timedelta(days=date)
    hour_int = int(hour // 3600)
    min_int = int(hour % 3600 // 60)
    sec_int = int(hour % 60)

    config["noise_house_prop"]["noise_mode"] = "no_noise"
    config["noise_hvac_prop"]["noise_mode"] = "no_noise"
    config["default_env_prop"]["cluster_prop"]["nb_agents"] = 1
    config["default_hvac_prop"]["cooling_capacity"] = HVAC_power
    config["default_env_prop"]["start_datetime_mode"] = "fixed"
    config["default_hvac_prop"]["lockout_duration"] = 1
    config["default_env_prop"]["start_datetime"] = str(
        datetime.datetime(date.year, date.month, date.day, hour_int, min_int, sec_int)
    )

    config["default_house_prop"]["Ua"] *= Ua_ratio
    config["default_house_prop"]["Cm"] *= Cm_ratio
    config["default_house_prop"]["Ca"] *= Ca_ratio
    config["default_house_prop"]["Hm"] *= Hm_ratio

    config["default_house_prop"]["init_air_temp"] = (
        config["default_house_prop"]["target_temp"] + air_temp
    )
    config["default_house_prop"]["init_mass_temp"] = (
        config["default_house_prop"]["target_temp"] + mass_temp
    )

    config["default_env_prop"]["cluster_prop"]["temp_mode"] = "constant"
    config["default_env_prop"]["cluster_prop"]["temp_parameters"]["constant"][
        "day_temp"
    ] = (config["default_house_prop"]["target_temp"] + OD_temp)
    config["default_env_prop"]["cluster_prop"]["temp_parameters"]["constant"][
        "night_temp"
    ] = (config["default_house_prop"]["target_temp"] + OD_temp)
    
    config["default_env_prop"]["power_grid_prop"]["base_power_mode"] = "constant"

    env = MADemandResponseEnv(config)

    houses = env.cluster.houses

    actors = {}
    for hvac_id in houses.keys():
        agent_prop = {"id": hvac_id}
        actors[hvac_id] = BangBangController(agent_prop, config)

    obs_dict = env.reset()
    for elem in obs_dict:
        obs_dict[elem]["reg_signal"] = 0

    total_cluster_hvac_power = 0

    actions = get_actions(actors, obs_dict)

    average_cluster_hvac_power = 0
    for i in range(NB_TIME_STEPS_BY_SIM):
        obs_dict, _, _, info = env.step(actions)
        total_cluster_hvac_power += info["cluster_hvac_power"]
        if i >= NB_TIME_STEPS_BY_SIM - NB_TIME_STEPS_AVG:
            average_cluster_hvac_power += total_cluster_hvac_power/((i+1) * NB_TIME_STEPS_AVG) # The average of the NB_TIME_STEPS_AVG last averages (to stabilize)
        actions = get_actions(actors, obs_dict)
        #print("Step {} - Avg power: {} - LastAvg: {} - Power: {}".format(i, total_cluster_hvac_power/(i+1), average, info["cluster_hvac_power"]))
    #average_cluster_hvac_power = total_cluster_hvac_power / NB_TIME_STEPS_BY_SIM
    return average_cluster_hvac_power


df = pd.DataFrame(columns=list(keys) + ["hvac_average_power"])
nb_combination_total = number_of_combination(parameters_dict)
start_time = time.time()
lowest_i = math.floor(opt.lower_fraction * nb_combination_total)
highest_i = math.ceil(opt.upper_fraction * nb_combination_total)

nb_combination_current_run = highest_i - lowest_i

print("Run from combination ", lowest_i, " to ", highest_i)

for i, parameters in enumerate(combinations):
    if i >= lowest_i and i <= highest_i:
        hvac_average_power = eval_parameters_bangbang_average_consumption(
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5],
            parameters[6],
            parameters[7],
            parameters[8],
            parameters[9],
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        [
                            parameters[0],
                            parameters[1],
                            parameters[2],
                            parameters[3],
                            parameters[4],
                            parameters[5],
                            parameters[6],
                            parameters[7],
                            parameters[8],
                            parameters[9],
                            hvac_average_power,
                        ]
                    ],
                    index=[i],
                    columns=list(keys) + ["hvac_average_power"],
                ),
            ],
        )

        if (i - lowest_i) % 500 == 0:
            print(
                "\nCombination: ",
                i - lowest_i,
                "/",
                nb_combination_current_run,
                "\nCompletion: ",
                round((i - lowest_i) / nb_combination_current_run * 100, 2),
                "%",
                "\nElapsed time since the beggining:",
                str(datetime.timedelta(seconds=round(time.time() - start_time))),
                "\nRemaining time estimation:",
                str(
                    datetime.timedelta(
                        seconds=(1 - (i - lowest_i) / nb_combination_current_run)
                        * (time.time() - start_time)
                        / (((i - lowest_i) + 1) / nb_combination_current_run)
                    )
                ),
            )
            if i % 500000 == 0:
                df.to_csv(
                    f"monteCarlo/gridSearchResult_backup_from_{lowest_i}_to_{i}.csv"
                )

df.to_csv(f"monteCarlo/gridSearchResultFinal_from_{lowest_i}_to_{highest_i}.csv")

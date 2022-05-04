import sys

sys.path.insert(1, "../marl-demandresponse")

import copy
import datetime
import itertools as it
import time

import pandas as pd
from agents import *
from config import config_dict
from env import *
from utils import get_actions

parameters_dict = {
    "Ua": [0.9, 1, 1.1],
    "Cm": [0.9, 1, 1.1],
    "Ca": [0.9, 1, 1.1],
    "Hm": [0.9, 1, 1.1],
    "air_temp": [-4, -2, -1, 0, 1, 2, 4],
    "mass_temp": [-4, -2, 0, 2, 4],
    "OD_temp": [3, 5, 7, 9, 11],
    "HVAC_power": [10000, 15000, 20000],
    "hour": [3, 6, 8, 11, 13, 16, 18, 21],
    "date": [(2021, 3, 21), (2021, 6, 21), (2021, 9, 21), (2021, 12, 21)],
}


keys = parameters_dict.keys()
combinations = it.product(*(parameters_dict[Name] for Name in keys))


def number_of_combination(dict):
    nb_comb = 1
    for key in dict:
        nb_comb *= len(dict[key])
    return nb_comb


def eval_parameters_bangbang_average_consumption(
    Ua, Cm, Ca, Hm, air_temp, mass_temp, OD_temp, HVAC_power, hour, date
):

    config = copy.deepcopy(config_dict)
    config["noise_house_prop"]["noise_mode"] = "no_noise"
    config["noise_hvac_prop"]["noise_mode"] = "no_noise"
    config["default_env_prop"]["cluster_prop"]["nb_agents"] = 1
    config["default_hvac_prop"]["cooling_capacity"] = HVAC_power
    config["default_env_prop"]["base_datetime"] = str(
        datetime.datetime(date[0], date[1], date[2], hour, 0, 0)
    )

    config["default_house_prop"]["Ua"] *= Ua
    config["default_house_prop"]["Cm"] *= Cm
    config["default_house_prop"]["Ca"] *= Ca
    config["default_house_prop"]["Hm"] *= Hm

    nb_time_steps = 450

    env = MADemandResponseEnv(config)

    hvacs_id_registry = env.cluster.hvacs_id_registry

    actors = {}
    for hvac_id in hvacs_id_registry.keys():
        agent_prop = {"id": hvac_id}
        actors[hvac_id] = DeadbandBangBangController(agent_prop, config)

    obs_dict = env.reset()
    for elem in obs_dict:
        obs_dict[elem]["OD_temp"] = obs_dict[elem]["house_target_temp"] + OD_temp
        obs_dict[elem]["house_temp"] = obs_dict[elem]["house_target_temp"] + air_temp
        obs_dict[elem]["house_mass_temp"] = (
            obs_dict[elem]["house_target_temp"] + mass_temp
        )
        env.start_datetime = datetime.datetime(date[0], date[1], date[2], hour, 0, 0)
        env.datetime = env.start_datetime
        obs_dict[elem]["reg_signal"] = 0

    total_cluster_hvac_power = 0

    actions = get_actions(actors, obs_dict)
    for i in range(nb_time_steps):

        obs_dict, _, _, info = env.step(actions)

        total_cluster_hvac_power += info["cluster_hvac_power"]

        for elem in obs_dict:
            obs_dict[elem]["OD_temp"] = obs_dict[elem]["house_target_temp"] + OD_temp
            obs_dict[elem]["datetime"] = datetime.datetime(
                date[0], date[1], date[2], hour, 0, 0
            )
        actions = get_actions(actors, obs_dict)
    average_cluster_hvac_power = total_cluster_hvac_power / nb_time_steps
    return average_cluster_hvac_power


df = pd.DataFrame(columns=list(keys) + ["hvac_average_power"])
nb_combination_total = number_of_combination(parameters_dict)
start_time = time.time()

for i, parameters in enumerate(combinations):
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
    df.loc[len(df.index)] = [
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
    if i % 500 == 0:
        print(
            "\nCombination: ",
            i,
            "/",
            nb_combination_total,
            "\nCompletion: ",
            round(i / nb_combination_total * 100, 2),
            "%",
            "\nElapsed time since the beggining:",
            str(datetime.timedelta(seconds=round(time.time() - start_time))),
            "\nRemaining time estimation:",
            str(
                datetime.timedelta(
                    seconds=(1 - i / nb_combination_total)
                    * (time.time() - start_time)
                    / ((i + 1) / nb_combination_total)
                )
            ),
        )
    if i % 50000 == 0:
        df.to_csv(f"monteCarlo/gridSearchResult{i}.csv")

df.to_csv("./gridSearchResultFinal.csv")

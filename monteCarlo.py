from env import *
from agents import *
from config import config_dict
from utils import get_actions
import datetime
import itertools as it
import pandas as pd


parameters_dict = {
    "Ua": [0.9, 1, 1.1],
    "Cm": [0.9, 1, 1.1],
    "Ca": [0.9, 1, 1.1],
    "Hm": [0.9, 1, 1.1],
    "air_temp": [-4, -2, -1, 0, 1, 2, 4],
    "mass_temp": [-4, -2, 0, 2, -4],
    "OD_temp": [3, 5, 7, 9, 11],
    "HVAC_power": [10000, 15000, 20000],
    "hour": [3, 6, 8, 11, 13, 16, 18, 21],
    "date": [(2021, 3, 21), (2021, 6, 21), (2021, 9, 21), (2021, 12, 21)],
}


keys = parameters_dict.keys()
combinations = it.product(*(parameters_dict[Name] for Name in keys))


def eval_parameters_bangbang_average_consumption(
    Ua, Cm, Ca, Hm, air_temp, mass_temp, OD_temp, HVAC_power, hour, date
):

    print(Ua, Cm, Ca, Hm, air_temp, mass_temp, OD_temp, HVAC_power, hour, date)
    config_dict["noise_house_prop"]["noise_mode"] = "no_noise"
    config_dict["noise_hvac_prop"]["noise_mode"] = "no_noise"
    config_dict["default_env_prop"]["cluster_prop"]["nb_agents"] = 1
    config_dict["default_hvac_prop"]["cooling_capacity"] = HVAC_power

    config_dict["default_house_prop"]["Ua"] *= Ua
    config_dict["default_house_prop"]["Cm"] *= Cm
    config_dict["default_house_prop"]["Ca"] *= Ca
    config_dict["default_house_prop"]["Hm"] *= Hm
    print(HVAC_power)
    nb_time_steps = 450

    env = MADemandResponseEnv(config_dict)

    hvacs_id_registry = env.cluster.hvacs_id_registry

    actors = {}
    for hvac_id in hvacs_id_registry.keys():
        agent_prop = {"id": hvac_id}
        actors[hvac_id] = DeadbandBangBangController(agent_prop, config_dict)

    # env.cluster.current_OD_temp = OD_temp +

    obs_dict = env.reset()

    for elem in obs_dict:
        obs_dict[elem]["OD_temp"] = obs_dict[elem]["house_target_temp"] + OD_temp
        obs_dict[elem]["house_temp"] = obs_dict[elem]["house_target_temp"] + air_temp
        obs_dict[elem]["house_mass_temp"] = (
            obs_dict[elem]["house_target_temp"] + mass_temp
        )
        obs_dict[elem]["datetime"] = datetime.datetime(
            date[0], date[1], date[2], hour, 0, 0
        )
        obs_dict[elem]["reg_signal"] = 0

    total_cluster_hvac_power = 0

    actions = get_actions(actors, obs_dict)
    for i in range(nb_time_steps):
        obs_dict, _, _, info = env.step(actions)
        for elem in obs_dict:
            obs_dict[elem]["OD_temp"] = 25
            obs_dict[elem]["datetime"] = datetime.datetime(2021, 2, 26, 9, 16, 13)
            obs_dict[elem]["reg_signal"] = 0
        actions = get_actions(actors, obs_dict)
        total_cluster_hvac_power += info["cluster_hvac_power"]

    average_cluster_hvac_power = total_cluster_hvac_power / nb_time_steps

    return average_cluster_hvac_power


df = pd.DataFrame(columns=list(keys) + ["hvac_average_power"])
print(df)

for parameters in combinations:
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

df.to_csv("./gridSearchResult.csv")

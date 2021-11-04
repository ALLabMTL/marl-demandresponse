import numpy as np
import random
from config import *


# Applying noise on properties
def apply_house_noise(house_prop, noise_house_prop):
    # Gaussian noise: target temp
    house_prop["target_temp"] += - np.abs(random.gauss(0, noise_house_prop["std_target_temp"]))

    # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance
    factor_Ua = random.triangular(noise_house_prop["factor_thermo_low"], noise_house_prop["factor_thermo_low"],
                                  1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ua"] *= factor_Ua

    factor_Cm = random.triangular(noise_house_prop["factor_thermo_low"], noise_house_prop["factor_thermo_low"],
                                  1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Cm"] *= factor_Cm

    factor_Ca = random.triangular(noise_house_prop["factor_thermo_low"], noise_house_prop["factor_thermo_low"],
                                  1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ca"] *= factor_Ca

    factor_Hm = random.triangular(noise_house_prop["factor_thermo_low"], noise_house_prop["factor_thermo_low"],
                                  1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Hm"] *= factor_Hm


def apply_hvac_noise(hvac_prop, noise_hvac_prop):
    # Gaussian noise: latent_cooling_fraction
    hvac_prop["latent_cooling_fraction"] += random.gauss(0, noise_hvac_prop["std_latent_cooling_fraction"])

    # Factor noise: COP, cooling_capacity
    factor_COP = random.triangular(noise_hvac_prop["factor_COP_low"], noise_hvac_prop["factor_COP_high"],
                                   1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    hvac_prop["COP"] *= factor_COP

    factor_cooling_capacity = random.triangular(noise_hvac_prop["factor_cooling_capacity_low"],
                                                noise_hvac_prop["factor_cooling_capacity_high"],
                                                1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    hvac_prop["cooling_capacity"] *= factor_cooling_capacity


# Multi agent management

def get_actions(actors, obs_dict):
    actions = {}
    for agent_id in actors.keys():
        actions[agent_id] = actors[agent_id].act(obs_dict[agent_id])
    return actions

def datetime2List(dt):
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute]

def superDict2List(SDict, id):
    tmp = SDict[id].copy()
    tmp['datetime'] = datetime2List(tmp['datetime'])
    for k,v in tmp.items():
        if not isinstance(tmp[k],list): tmp[k] = [v]
    return sum(list(tmp.values()), [])

def normSuperDict(sDict, id, returnDict=False):
    result = {}
    k_temp = ['OD_temp','house_temp','house_target_temp']
    k_div = ['house_Ua','house_Cm','house_Ca','house_Hm','hvac_COP','hvac_cooling_capacity','hvac_latent_cooling_fraction']
    # k_lockdown = ['hvac_seconds_since_off', 'hvac_lockout_duration']
    for k in k_temp:
        result[k] = (sDict[id][k]-default_house_prop["target_temp"])/noise_house_prop["std_target_temp"]
    result["house_deadband"] = sDict[id]["house_deadband"]/noise_house_prop["std_target_temp"]
    day = sDict[id]['datetime'].timetuple().tm_yday
    hour = sDict[id]['datetime'].hour
    result["sin_day"] = (np.sin(day*2*np.pi/365))
    result["cos_day"] = (np.cos(day*2*np.pi/365))
    result["sin_hr"] = np.sin(hour*2*np.pi/24)
    result["cos_hr"] = np.cos(hour*2*np.pi/24)
    for k in k_div:
        k1 = "_".join(k.split("_")[1:])
        if k1 in list(default_house_prop.keys()):
            result[k] = sDict[id][k]/default_house_prop[k1]
        elif k1 in list(default_hvac_prop.keys()):
            result[k] = sDict[id][k]/default_hvac_prop[k1]
        else:
            print(k)
            raise Exception("Error Key Matching.")
    result["hvac_turned_on"] = sDict[id]["hvac_turned_on"]
    result["hvac_seconds_since_off"] = sDict[id]["hvac_seconds_since_off"]/sDict[id]["hvac_lockout_duration"]
    result["hvac_lockout_duration"] = sDict[id]["hvac_lockout_duration"]/sDict[id]["hvac_lockout_duration"]
    return result if returnDict else list(result.values())
import numpy as np
import random


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
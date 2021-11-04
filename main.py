from env import *
from agents import *
from config import default_house_prop, noise_house_prop, default_hvac_prop, noise_hvac_prop, default_env_properties
from utils import apply_house_noise, apply_hvac_noise, get_actions

from copy import deepcopy
import warnings
import random
import numpy as np

nb_houses = 100
num_steps = 2000

# random.seed(1)


# Creating houses
houses_properties = []
agent_ids = []
for i in range(nb_houses):
    house_prop = deepcopy(default_house_prop)
    apply_house_noise(house_prop, noise_house_prop)
    house_prop["id"] = str(i)
    hvac_prop = deepcopy(default_hvac_prop)
    apply_hvac_noise(hvac_prop, noise_hvac_prop)
    hvac_id = str(i) + "_1"
    hvac_prop["id"] = hvac_id
    agent_ids.append(hvac_id)
    house_prop["hvac_properties"] = [hvac_prop]
    houses_properties.append(house_prop)

# Setting environment properties
env_properties = deepcopy(default_env_properties)
env_properties["cluster_properties"]["houses_properties"] = houses_properties
env_properties["agent_ids"] = agent_ids
env_properties["nb_hvac"] = len(agent_ids)

env = MADemandResponseEnv(env_properties)
hvacs_id_registry = env.cluster.hvacs_id_registry

actors = {}
for hvac_id in hvacs_id_registry.keys():
    agent_prop = {"id": hvac_id}

    # actors[hvac_id] = DeadbandBangBangController(agent_prop)
    actors[hvac_id] = DeadbandBangBangController(agent_prop)

obs = env.reset()

total_cluster_hvac_power = 0
for i in range(num_steps):
    actions = get_actions(actors, obs)
    obs, _, _, info = env.step(actions)
    total_cluster_hvac_power += info["cluster_hvac_power"]

average_cluster_hvac_power = total_cluster_hvac_power / num_steps
average_hvac_power = average_cluster_hvac_power / nb_houses
print("Average cluster hvac power: {:f} W, per hvac: {:f} W".format(average_cluster_hvac_power, average_hvac_power))


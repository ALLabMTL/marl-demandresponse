from env import *
from agents import *
from config import config_dict
from utils import get_actions

from copy import deepcopy
import warnings
import os
import random
import numpy as np

import argparse
import wandb


os.environ["WANDB_SILENT"] = "true"

agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "PPO": PPOAgent,
    "GreedyMyopic": GreedyMyopic
}


# CLI arguments

parser = argparse.ArgumentParser(description="Deployment options")

parser.add_argument(
    "--agent",
    type=str,
    choices=agents_dict.keys(),
    required=True,
    help="Agent for control"
)

parser.add_argument(
    "--nb_agents",
    type=int,
    default=1,
    help="Number of agents (TCLs)",
)

parser.add_argument(
    "--nb_time_steps",
    type=int,
    default=1000,
    help="Number of time steps in an episode",
)

parser.add_argument(
    "--env_seed",
    type=int,
    default=1,
    help="Environment seed",
)

parser.add_argument(
    "--net_seed",
    type=int,
    default=1,
    help="Network and torch seed",
)

parser.add_argument(
    "--exp",
    type=str,
    default="Deploy",
    help="Experiment name",
)

parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Add to prevent logging to wandb",
)

parser.add_argument(
    "--render",
    action="store_true",
    help="Add to generate a visual render of the simulation",
)

parser.add_argument(
    "--cooling_capacity",
    type=int,
    default=-1,
    help="Default cooling capacity of the HVACs",
)

parser.add_argument(
    "--time_step",
    type=int,
    default=-1,
    help="Time step in seconds",
)

parser.add_argument(
    "--lockout_duration",
    type=int,
    default=-1,
    help="Default AC lockout duration, in seconds",
)

parser.add_argument(
    "--actor_name",
    type=str,
    default=None,
    help="Name of the trained agent to load")

parser.add_argument(
    "--exploration_temp",
    type=float,
    default=1.0,
    help="Temperature of the policy softmax. Higher temp -> more exploration.")

parser.add_argument(
    "--house_noise_mode",
    type=str,
    default="config",
    help="Mode of noise over house parameters.")

parser.add_argument(
    "--hvac_noise_mode",
    type=str,
    default="config",
    help="Mode of noise over hvac parameters.")

parser.add_argument(
    "--OD_temp_mode",
    type=str,
    default="config",
    help="Mode of outdoors temperature.")

parser.add_argument(
    "--no_solar_gain",
    action="store_true",
    help="Removes the solar gain from the simulation.")

opt = parser.parse_args()

if opt.render:
    from env.renderer import Renderer
    renderer = Renderer(opt.nb_agents)

# Creating environment
random.seed(opt.env_seed)

if opt.nb_agents != -1:
    config_dict["default_env_prop"]["cluster_prop"]["nb_agents"] = opt.nb_agents
if opt.time_step != -1:
    config_dict["default_env_prop"]['time_step'] = opt.time_step
if opt.cooling_capacity != -1:
    config_dict["default_hvac_prop"]['cooling_capacity'] = opt.cooling_capacity
if opt.lockout_duration != -1:
    config_dict["default_hvac_prop"]['lockout_duration'] = opt.lockout_duration
if opt.house_noise_mode != "config":
    config_dict["noise_house_prop"]['noise_mode'] = opt.house_noise_mode
if opt.hvac_noise_mode != "config":
    config_dict["noise_hvac_prop"]['noise_mode'] = opt.hvac_noise_mode
if opt.OD_temp_mode != "config":
    config_dict["default_env_prop"]['cluster_prop']["temp_mode"] = opt.OD_temp_mode
if opt.no_solar_gain:
    config_dict["default_house_prop"]["shading_coeff"] = 0

nb_time_steps = opt.nb_time_steps


env = MADemandResponseEnv(config_dict)
nb_agents = config_dict["default_env_prop"]["cluster_prop"]["nb_agents"]
hvacs_id_registry = env.cluster.hvacs_id_registry

actors = {}
for hvac_id in hvacs_id_registry.keys():
    agent_prop = {"id": hvac_id}

    if opt.actor_name:
        agent_prop["actor_name"] = opt.actor_name
        agent_prop["net_seed"] = opt.net_seed
        agent_prop["exploration_temp"] = opt.exploration_temp

    actors[hvac_id] = agents_dict[opt.agent](agent_prop, config_dict)


obs_dict = env.reset()

total_cluster_hvac_power = 0
on_off_ratio = 0

for i in range(nb_time_steps):
    actions = get_actions(actors, obs_dict)
    obs_dict, _, _, info = env.step(actions)
    if opt.render:
        renderer.render(obs_dict)
    total_cluster_hvac_power += info["cluster_hvac_power"]
    for k in actions.keys():
        if actions[k]:
            on_off_ratio += 1./len(actions.keys())


average_cluster_hvac_power = total_cluster_hvac_power / nb_time_steps
average_hvac_power = average_cluster_hvac_power / nb_agents
on_off_timeratio = on_off_ratio / nb_time_steps
print("Average cluster hvac power: {:f} W, per hvac: {:f} W".format(
    average_cluster_hvac_power, average_hvac_power))
print("On_off time ratio: {}".format(on_off_timeratio))

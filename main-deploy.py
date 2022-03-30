#from apt import ProblemResolver
from env import *
from agents import *
from config import config_dict
from utils import get_actions
from wandb_setup import wandb_setup
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
    "--nb_logs",
    type=int,
    default=100,
    help="Number of logging points for training stats",
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
    "--signal_mode",
    type=str,
    default="config",
    help="Mode of power grid regulation signal simulation."
)

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
log_wandb = not opt.no_wandb


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
if opt.signal_mode != "config":
    config_dict["default_env_prop"]['power_grid_prop']["signal_mode"] = opt.signal_mode
if opt.house_noise_mode != "config":
    config_dict["noise_house_prop"]['noise_mode'] = opt.house_noise_mode
if opt.hvac_noise_mode != "config":
    config_dict["noise_hvac_prop"]['noise_mode'] = opt.hvac_noise_mode
if opt.OD_temp_mode != "config":
    config_dict["default_env_prop"]['cluster_prop']["temp_mode"] = opt.OD_temp_mode
if opt.no_solar_gain:
    config_dict["default_house_prop"]["shading_coeff"] = 0
if log_wandb:
    wandb_run = wandb_setup(opt, config_dict)

nb_time_steps = opt.nb_time_steps


env = MADemandResponseEnv(config_dict)
time_steps_log = int(opt.nb_time_steps/opt.nb_logs)
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

cumul_temp_offset = 0
cumul_temp_error = 0
cumul_signal_offset = 0
cumul_signal_error = 0
actions = get_actions(actors, obs_dict)
for i in range(nb_time_steps):
    obs_dict, _, _, info = env.step(actions)
    actions = get_actions(actors, obs_dict)
    if opt.render:
        renderer.render(obs_dict)
    total_cluster_hvac_power += info["cluster_hvac_power"]
    for k in actions.keys():
        if actions[k]:
            on_off_ratio += 1./len(actions.keys())

    for k in obs_dict.keys():
        cumul_temp_offset += (obs_dict[k]["house_temp"] -
                              obs_dict[k]["house_target_temp"]) / env.nb_agents
        cumul_temp_error += np.abs(obs_dict[k]["house_temp"] -
                                   obs_dict[k]["house_target_temp"]) / env.nb_agents
    cumul_signal_offset += obs_dict['0_1']["reg_signal"] - \
        obs_dict['0_1']["cluster_hvac_power"]
    cumul_signal_error += np.abs(
        obs_dict['0_1']["reg_signal"] - obs_dict['0_1']["cluster_hvac_power"])

    if i % time_steps_log == time_steps_log - 1:       # Log train statistics
        #print("Logging stats at time {}".format(t))

        mean_temp_offset = cumul_temp_offset/time_steps_log
        mean_temp_error = cumul_temp_error/time_steps_log
        mean_signal_offset = cumul_signal_offset/time_steps_log
        mean_signal_error = cumul_signal_error/time_steps_log

        if log_wandb:
            wandb_run.log({"Mean temperature offset": mean_temp_offset, "Mean temperature error": mean_temp_error,
                           "Mean signal offset": mean_signal_offset, "Mean signal error": mean_signal_error, "Time step": i})

        cumul_temp_offset = 0
        cumul_temp_error = 0
        cumul_signal_offset = 0
        cumul_signal_error = 0
average_cluster_hvac_power = total_cluster_hvac_power / nb_time_steps
average_hvac_power = average_cluster_hvac_power / nb_agents
on_off_timeratio = on_off_ratio / nb_time_steps
print("Average cluster hvac power: {:f} W, per hvac: {:f} W".format(
    average_cluster_hvac_power, average_hvac_power))
print("On_off time ratio: {}".format(on_off_timeratio))

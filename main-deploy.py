# from apt import ProblemResolver
from env import *
from agents import *
from config import config_dict
from utils import get_actions, adjust_config_deploy, normStateDict
from wandb_setup import wandb_setup
from copy import deepcopy
import warnings
import os
import random
import numpy as np

import argparse
import wandb
from cli import cli_deploy


os.environ["WANDB_SILENT"] = "true"

agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "PPO": PPOAgent,
    "GreedyMyopic": GreedyMyopic,
}


# CLI arguments

opt = cli_deploy(agents_dict)
adjust_config_deploy(opt, config_dict)

log_wandb = not opt.no_wandb
if opt.render:
    from env.renderer import Renderer

    renderer = Renderer(opt.nb_agents)

# Creating environment
random.seed(opt.env_seed)
nb_time_steps = opt.nb_time_steps

if log_wandb:
    wandb_run = wandb_setup(opt, config_dict)

env = MADemandResponseEnv(config_dict)
obs_dict = env.reset()
num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))


time_steps_log = int(opt.nb_time_steps / opt.nb_logs)
nb_agents = config_dict["default_env_prop"]["cluster_prop"]["nb_agents"]
houses = env.cluster.houses

actors = {}
for house_id in houses.keys():
    agent_prop = {"id": house_id}

    if opt.actor_name:
        agent_prop["actor_name"] = opt.actor_name
        agent_prop["net_seed"] = opt.net_seed
        agent_prop["exploration_temp"] = opt.exploration_temp

    actors[house_id] = agents_dict[opt.agent](agent_prop, config_dict, num_state=num_state)


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
            on_off_ratio += 1.0 / len(actions.keys())

    for k in obs_dict.keys():
        cumul_temp_offset += (
            obs_dict[k]["house_temp"] - obs_dict[k]["house_target_temp"]
        ) / env.nb_agents
        cumul_temp_error += (
            np.abs(obs_dict[k]["house_temp"] - obs_dict[k]["house_target_temp"])
            / env.nb_agents
        )
    cumul_signal_offset += (
        obs_dict[0]["reg_signal"] - obs_dict[0]["cluster_hvac_power"]
    )
    cumul_signal_error += np.abs(
        obs_dict[0]["reg_signal"] - obs_dict[0]["cluster_hvac_power"]
    )

    if i % time_steps_log == time_steps_log - 1:  # Log train statistics
        # print("Logging stats at time {}".format(t))

        mean_temp_offset = cumul_temp_offset / time_steps_log
        mean_temp_error = cumul_temp_error / time_steps_log
        mean_signal_offset = cumul_signal_offset / time_steps_log
        mean_signal_error = cumul_signal_error / time_steps_log

        if log_wandb:
            wandb_run.log(
                {
                    "Mean temperature offset": mean_temp_offset,
                    "Mean temperature error": mean_temp_error,
                    "Mean signal offset": mean_signal_offset,
                    "Mean signal error": mean_signal_error,
                    "Time step": i,
                }
            )

        cumul_temp_offset = 0
        cumul_temp_error = 0
        cumul_signal_offset = 0
        cumul_signal_error = 0
average_cluster_hvac_power = total_cluster_hvac_power / nb_time_steps
average_hvac_power = average_cluster_hvac_power / nb_agents
on_off_timeratio = on_off_ratio / nb_time_steps
print(
    "Average cluster hvac power: {:f} W, per hvac: {:f} W".format(
        average_cluster_hvac_power, average_hvac_power
    )
)
print("On_off time ratio: {}".format(on_off_timeratio))

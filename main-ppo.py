from env import *
from agents import *
from config import (
    default_house_prop,
    noise_house_prop,
    default_hvac_prop,
    noise_hvac_prop,
    default_env_properties,
)
from utils import apply_house_noise, apply_hvac_noise, get_actions

import argparse

import wandb
import os

from copy import deepcopy
import warnings
import random
import numpy as np
from collections import namedtuple
from itertools import count
from agents.ppo import PPO
from env.MA_DemandResponse import MADemandResponseEnv as env
from utils import (
    normSuperDict,
    normStateDict,
    testAgentHouseTemperature,
    colorPlotTestAgentHouseTemp,
    saveActorNetDict,
)
from wandb_setup import wandb_setup


os.environ["WANDB_SILENT"] = "true"


# CLI arguments

parser = argparse.ArgumentParser(description="Training options")

parser.add_argument(
    "--nb_agents", 
    type=int, 
    default=-1, 
    help="Number of agents (TCLs)",
)

parser.add_argument(
    "--nb_tr_episodes", 
    type=int, 
    default=1000, 
    help="Number of episodes for training",
)

parser.add_argument(
    "--nb_time_steps", 
    type=int, 
    default=1000, 
    help="Total number of time steps",
)

parser.add_argument(
    "--ppo_bs", 
    type=int, 
    default=32, 
    help="Batch size of PPO",
)

parser.add_argument(
    "--net_seed", 
    type=int, 
    default=1, 
    help="Neural network seed",
)

parser.add_argument(
    "--env_seed", 
    type=int, 
    default=1, 
    help="Environment seed",
)

parser.add_argument(
    "--exp", 
    type=str, 
    required=True, 
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
    "--save_actor_name",
    type=str,
    default=None,
    help="Name to store the actor agent after training",
)

parser.add_argument(
    "--exploration_temp",
    type=float,
    default=1.0,
    help="Temperature of the policy softmax. Higher temp -> more exploration.")

opt = parser.parse_args()

# Renderer import
if opt.render:
    from env.renderer import Renderer
render = opt.render

# Starting WandB
log_wandb = not opt.no_wandb

if log_wandb:
    wandb_run = wandb_setup(opt)


# Creating environment
random.seed(opt.env_seed)

if opt.nb_agents != -1:
    default_env_properties["cluster_properties"]["nb_agents"] = opt.nb_agents
if opt.time_step != -1:
    default_env_properties['time_step'] = opt.time_step
if opt.cooling_capacity != -1:
    default_hvac_prop['cooling_capacity'] = opt.cooling_capacity
if opt.lockout_duration != -1:
    default_hvac_prop['lockout_duration'] = opt.lockout_duration

env = MADemandResponseEnv(default_env_properties, default_house_prop, noise_house_prop, default_hvac_prop, noise_hvac_prop)
nb_agents = default_env_properties["cluster_properties"]["nb_agents"]
hvacs_id_registry = env.cluster.hvacs_id_registry


time_steps_per_ep = int(opt.nb_time_steps/opt.nb_tr_episodes)

## Training loop
if __name__ == "__main__":
    Transition = namedtuple(
        "Transition", ["state", "action", "a_log_prob", "reward", "next_state"]
    )
    agent = PPO(seed=opt.net_seed, bs=opt.ppo_bs, log_wandb=log_wandb)
    if render:
        renderer = Renderer(nb_agents)
    prob_on_episode = np.empty(100)
    for episode in range(opt.nb_tr_episodes):

        obs_dict = env.reset()

        mean_return = 0
        mean_temp_offset = 0
        for t in range(time_steps_per_ep):
            action_and_prob = {
                k: agent.select_action(normStateDict(obs_dict[k]), temp=opt.exploration_temp)
                for k in obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
            action_prob = {k: action_and_prob[k][1] for k in obs_dict.keys()}
            next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            mean_reward = 0
            cumul_temp_offset = 0
            for k in obs_dict.keys():
                agent.store_transition(
                    Transition(
                        normStateDict(obs_dict[k]),
                        action[k],
                        action_prob[k],
                        rewards_dict[k],
                        normStateDict(next_obs_dict[k]),
                    )
                )
                mean_reward += rewards_dict[k] / nb_agents
                cumul_temp_offset += (next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / nb_agents

            obs_dict = next_obs_dict
            mean_return += float(mean_reward) / time_steps_per_ep
            mean_temp_offset += cumul_temp_offset / time_steps_per_ep
            if render:
                renderer.render(obs_dict)

        if log_wandb:
            wandb_run.log({"Mean train return": mean_return, "Mean temperature offset": mean_temp_offset, "Training steps": time_steps_per_ep*episode + t})
        if len(agent.buffer) >= agent.batch_size:

            agent.update(episode)
        prob_on = testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30)
        prob_on_episode = np.vstack((prob_on_episode, testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30)))
    prob_on_episode = prob_on_episode[1:]

    colorPlotTestAgentHouseTemp(prob_on_episode, 10, 30, log_wandb)

    if opt.save_actor_name:
        path = os.path.join(".","actors", opt.save_actor_name) 
        saveActorNetDict(agent, path)

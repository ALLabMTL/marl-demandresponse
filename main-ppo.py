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
)


os.environ["WANDB_SILENT"] = "true"


# CLI arguments

parser = argparse.ArgumentParser(description="Training options")

parser.add_argument(
    "--nb_agents", 
    type=int, 
    default=1, 
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
    help="Number of time steps in an episode",
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

opt = parser.parse_args()

if opt.render:
    from env.renderer import Renderer



log_wandb = not opt.no_wandb
render = opt.render
# Starting WandB
if log_wandb:
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="ProofConcept",
        entity="marl-dr",
        config=opt,
        name="%s_TCLs-%d_envseed-%d_netseed-%d"
        % (opt.exp, opt.nb_agents, opt.env_seed, opt.net_seed),
    )


# Creating environment
random.seed(opt.env_seed)

if opt.time_step != -1:
    default_env_properties['time_step'] = opt.time_step
if opt.cooling_capacity != -1:
    default_hvac_prop['cooling_capacity'] = opt.cooling_capacity
if opt.lockout_duration != -1:
    default_hvac_prop['lockout_duration'] = opt.lockout_duration


## Creating houses
houses_properties = []
agent_ids = []
for i in range(opt.nb_agents):
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

## Setting environment properties
env_properties = deepcopy(default_env_properties)
env_properties["cluster_properties"]["houses_properties"] = houses_properties
env_properties["agent_ids"] = agent_ids
env_properties["nb_hvac"] = len(agent_ids)

env = MADemandResponseEnv(env_properties)
hvacs_id_registry = env.cluster.hvacs_id_registry


## Training loop
if __name__ == "__main__":
    Transition = namedtuple(
        "Transition", ["state", "action", "a_log_prob", "reward", "next_state"]
    )
    agent = PPO(seed=opt.net_seed, bs=opt.ppo_bs, log_wandb=log_wandb)
    temp = 1.1
    if render:
        renderer = Renderer(opt.nb_agents)
    prob_on_episode = np.empty(100)
    for episode in range(opt.nb_tr_episodes):

        obs_dict = env.reset()

        mean_return = 0
        for t in range(opt.nb_time_steps):

            action_and_prob = {
                k: agent.select_action(normStateDict(obs_dict[k]), temp=temp)
                for k in obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
            action_prob = {k: action_and_prob[k][1] for k in obs_dict.keys()}
            next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            mean_reward = 0
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
                mean_reward += rewards_dict[k] / opt.nb_agents

            obs_dict = next_obs_dict
            mean_return += float(mean_reward) / opt.nb_time_steps
            if render:
                renderer.render(obs_dict)

        if log_wandb:
            wandb.log({"Mean return": mean_return})
        if len(agent.buffer) >= agent.batch_size:

            agent.update(episode)
        prob_on = testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30)
        prob_on_episode = np.vstack((prob_on_episode, testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30)))
    prob_on_episode = prob_on_episode[1:]
    colorPlotTestAgentHouseTemp(prob_on_episode, 10, 30, log_wandb)

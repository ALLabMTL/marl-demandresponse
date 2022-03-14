from env import *
from agents import *
from config import config_dict
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
    help="Number of episodes (environment resets) for training",
)

parser.add_argument(
    "--nb_tr_epochs",
    type=int,
    default=20,
    help="Number of epochs (policy updates) for training",
)

parser.add_argument(
    "--nb_tr_logs",
    type=int,
    default=100,
    help="Number of logging points for training stats",
)

parser.add_argument(
    "--nb_test_logs",
    type=int,
    default=100,
    help="Number of logging points for testing stats (and thus, testing sessions)",
)

parser.add_argument(
    "--nb_time_steps",
    type=int,
    default=10000,
    help="Total number of time steps",
)

parser.add_argument(
    "--nb_time_steps_test",
    type=int,
    default=50000,
    help="Total number of time steps in an episode at test time",
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
    "--render_after",
    type=int,
    default=-1,
    help="Delay in time steps before rendering")

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
    help="Temperature of the policy softmax. Higher temp -> more exploration."
)

parser.add_argument(
    "--signal_mode",
    type=str,
    default="config",
    help="Mode of power grid regulation signal simulation."
)

parser.add_argument(
    "--alpha",
    type=float,
    default=-1,
    help="Tradeoff parameter for loss function: temperature penalty + alpha * regulation signal penalty."
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

# Renderer import
if opt.render:
    from env.renderer import Renderer
render = opt.render

# Starting WandB
log_wandb = not opt.no_wandb


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
if opt.alpha != -1:
    config_dict["default_env_prop"]["alpha"] = opt.alpha
if log_wandb:
    wandb_run = wandb_setup(opt, config_dict)

env = MADemandResponseEnv(config_dict)

time_steps_per_episode = int(opt.nb_time_steps/opt.nb_tr_episodes)
time_steps_per_epoch = int(opt.nb_time_steps/opt.nb_tr_epochs)
time_steps_train_log = int(opt.nb_time_steps/opt.nb_tr_logs)
time_steps_test_log = int(opt.nb_time_steps/opt.nb_test_logs)


def testAgent(agent, env, nb_time_steps_test):
    """
    Test agent on an episode of nb_test_timesteps, with 
    """

    cumul_avg_reward = 0

    obs_dict = env.reset()

    for t in range(nb_time_steps_test):
        action_and_prob = {k: agent.select_action(normStateDict(
            obs_dict[k], config_dict), temp=opt.exploration_temp) for k in obs_dict.keys()}
        action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
        obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        cumul_avg_reward += rewards_dict[k] / env.nb_agents

    mean_avg_return = cumul_avg_reward/nb_time_steps_test

    return mean_avg_return


# Training loop
if __name__ == "__main__":
    Transition = namedtuple(
        "Transition", ["state", "action", "a_log_prob", "reward", "next_state"])
    if render:
        renderer = Renderer(env.nb_agents)
    prob_on_test = np.empty(100)

    obs_dict = env.reset()
    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))

    agent = PPO(seed=opt.net_seed, bs=opt.ppo_bs,
                log_wandb=log_wandb, num_state=num_state)

    cumul_avg_reward = 0
    cumul_temp_offset = 0
    cumul_temp_error = 0
    cumul_signal_offset = 0
    cumul_signal_error = 0

    for t in range(opt.nb_time_steps):
        if render:
            renderer.render(obs_dict)
        # Taking action in environment
        action_and_prob = {k: agent.select_action(normStateDict(
            obs_dict[k], config_dict), temp=opt.exploration_temp) for k in obs_dict.keys()}
        action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
        action_prob = {k: action_and_prob[k][1] for k in obs_dict.keys()}
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        if render and t >= opt.render_after:
            renderer.render(next_obs_dict)


        # Storing in replay buffer
        for k in obs_dict.keys():
            agent.store_transition(Transition(normStateDict(
                obs_dict[k], config_dict), action[k], action_prob[k], rewards_dict[k], normStateDict(next_obs_dict[k], config_dict)))
            cumul_temp_offset += (next_obs_dict[k]["house_temp"] -
                                  next_obs_dict[k]["house_target_temp"]) / env.nb_agents
            cumul_temp_error += np.abs(next_obs_dict[k]["house_temp"] -
                                       next_obs_dict[k]["house_target_temp"]) / env.nb_agents
            cumul_avg_reward += rewards_dict[k] / env.nb_agents

        obs_dict = next_obs_dict

        # Mean values
        cumul_signal_offset += next_obs_dict['0_1']["reg_signal"] - \
            next_obs_dict['0_1']["cluster_hvac_power"]
        cumul_signal_error += np.abs(
            next_obs_dict['0_1']["reg_signal"] - next_obs_dict['0_1']["cluster_hvac_power"])
        #print(next_obs_dict['0_1']["reg_signal"] - next_obs_dict['0_1']["cluster_hvac_power"])

        if t % time_steps_per_episode == time_steps_per_episode - 1:     # Episode: reset environment
            print("New episode at time {}".format(t))
            obs_dict = env.reset()

        # Epoch: update agent
        if t % time_steps_per_epoch == time_steps_per_epoch - 1 and len(agent.buffer) >= agent.batch_size:
            print("Updating agent at time {}".format(t))

            agent.update(t)

        if t % time_steps_train_log == time_steps_train_log - 1:       # Log train statistics
            #print("Logging stats at time {}".format(t))

            mean_avg_return = cumul_avg_reward/time_steps_train_log
            mean_temp_offset = cumul_temp_offset/time_steps_train_log
            mean_temp_error = cumul_temp_error/time_steps_train_log
            mean_signal_offset = cumul_signal_offset/time_steps_train_log
            mean_signal_error = cumul_signal_error/time_steps_train_log

            if log_wandb:
                wandb_run.log({"Mean train return": mean_avg_return, "Mean temperature offset": mean_temp_offset, "Mean temperature error": mean_temp_error,
                              "Mean signal offset": mean_signal_offset, "Mean signal error": mean_signal_error, "Training steps": t})

            cumul_temp_offset = 0
            cumul_temp_error = 0
            cumul_avg_reward = 0
            cumul_signal_offset = 0
            cumul_signal_error = 0

        if t % time_steps_test_log == time_steps_test_log - 1:        # Test policy
            print("Testing at time {}".format(t))
            prob_on_test = np.vstack((prob_on_test, testAgentHouseTemperature(
                agent, obs_dict["0_1"], 10, 30, config_dict)))
            random.seed(t)
            test_env = MADemandResponseEnv(config_dict, test=True)

            mean_test_return = testAgent(
                agent, test_env, opt.nb_time_steps_test)
            if log_wandb:
                wandb_run.log(
                    {"Mean test return": mean_test_return, "Training steps": t})
            else:
                print(
                    "Training step - {} - Mean test return: {}".format(t, mean_test_return))

    if render:

        renderer.__del__(obs_dict)
    prob_on_test = prob_on_test[1:]

    colorPlotTestAgentHouseTemp(
        prob_on_test, 10, 30, time_steps_test_log, log_wandb)

    if opt.save_actor_name:
        path = os.path.join(".", "actors", opt.save_actor_name)
        saveActorNetDict(agent, path)


'''


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
        for t in range(time_steps_per_episode):
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
            mean_return += float(mean_reward) / time_steps_per_episode
            mean_temp_offset += cumul_temp_offset / time_steps_per_episode
            if render:
                renderer.render(obs_dict)

        if log_wandb:
            wandb_run.log({"Mean train return": mean_return, "Mean temperature offset": mean_temp_offset, "Training steps": time_steps_per_episode*episode + t})
        
        if len(agent.buffer) >= agent.batch_size:
            agent.update(episode)
        prob_on = testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30)
        prob_on_episode = np.vstack((prob_on_episode, testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30)))
    prob_on_episode = prob_on_episode[1:]

    colorPlotTestAgentHouseTemp(prob_on_episode, 10, 30, log_wandb)

    if opt.save_actor_name:
        path = os.path.join(".","actors", opt.save_actor_name) 
        saveActorNetDict(agent, path)
'''

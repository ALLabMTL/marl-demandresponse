#%% Imports

from env import *
from agents import *
from config import config_dict
from cli import cli_train
from agents.ppo import PPO
from env.MA_DemandResponse import MADemandResponseEnv as env
from utils import (
    normStateDict,
    testAgentHouseTemperature,
    colorPlotTestAgentHouseTemp,
    saveActorNetDict,
    adjust_config,
    render_and_wandb_init,
)

import os
import torch
from copy import deepcopy
import random
import numpy as np
from collections import namedtuple

os.environ["WANDB_SILENT"] = "true"

#%% Initializing

opt = cli_train() # get arguments from cli
adjust_config(opt, config_dict)
render, log_wandb, wandb_run = render_and_wandb_init(opt, config_dict)

# Creating environment
random.seed(opt.env_seed)
env = MADemandResponseEnv(config_dict)

if render:
    from env.renderer import Renderer
    renderer = Renderer(env.nb_agents)

#%% Variables and functions

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
    with torch.no_grad():
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
    prob_on_test_on = np.empty(100)
    prob_on_test_off = np.empty(100)

    obs_dict = env.reset()
    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))

    agent = PPO(seed=opt.net_seed, bs=opt.batch_size,
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
        cumul_signal_offset += next_obs_dict['0_1']["reg_signal"] - next_obs_dict['0_1']["cluster_hvac_power"]
        cumul_signal_error += np.abs(next_obs_dict['0_1']["reg_signal"] - next_obs_dict['0_1']["cluster_hvac_power"])
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
            prob_on_test_on = np.vstack((prob_on_test_on, testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30, config_dict, obs_dict["0_1"]["hvac_cooling_capacity"]/obs_dict["0_1"]["hvac_COP"])))
            prob_on_test_off = np.vstack((prob_on_test_off, testAgentHouseTemperature(agent, obs_dict["0_1"], 10, 30, config_dict, 0.0)))

            # random.seed(t)
            test_env = deepcopy(env)

            #test_env = MADemandResponseEnv(config_dict, test=True)

            mean_test_return = testAgent(agent, test_env, opt.nb_time_steps_test)
            if log_wandb:
                wandb_run.log(
                    {"Mean test return": mean_test_return, "Training steps": t})
            else:
                print(
                    "Training step - {} - Mean test return: {}".format(t, mean_test_return))

    if render:
        renderer.__del__(obs_dict)

    prob_on_test_on = prob_on_test_on[1:]
    prob_on_test_off = prob_on_test_off[1:]

    colorPlotTestAgentHouseTemp(prob_on_test_on, prob_on_test_off, 10, 30, time_steps_test_log, log_wandb)

    if opt.save_actor_name:
        path = os.path.join(".", "actors", opt.save_actor_name)
        saveActorNetDict(agent, path)
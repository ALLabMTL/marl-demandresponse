#%% Imports

from env import *
from agents import *
from config import config_dict
from cli import cli_train
from agents.dqn import DQN
from env.MA_DemandResponse import MADemandResponseEnv as env
from utils import (
    normStateDict,
    adjust_config,
    colorPlotTestAgentHouseTemp,
    render_and_wandb_init,
)

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

os.environ["WANDB_SILENT"] = "true"

#%% Initializing

opt = cli_train() # get arguments from cli
# opt.no_wandb = True
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
    Test agent on an episode of nb_test_timesteps
    """

    cumul_avg_reward = 0

    obs_dict = env.reset()

    for t in range(nb_time_steps_test):
        action = {k: agent.select_action(normStateDict(
            obs_dict[k], config_dict)) for k in obs_dict.keys()}
        obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        cumul_avg_reward += rewards_dict[k] / env.nb_agents
        
        if t % time_steps_per_episode == time_steps_per_episode - 1:
            print("New episode at time {}".format(t))
            obs_dict = env.reset()

    mean_avg_return = cumul_avg_reward/nb_time_steps_test

    return mean_avg_return

def get_agent_test(agent, state, low_temp, high_temp, config_dict, reg_signal):
    '''
    Receives an agent and a given state. Tests the agent output for 100 points a given range of indoors temperature, returning a vector of actions.
    '''
    temp_range = np.linspace(low_temp, high_temp, num=100)
    actions = np.zeros(100)
    for i in range(100):
        temp = temp_range[i]
        state['house_temp'] = temp
        state['reg_signal'] = reg_signal
        norm_state = normStateDict(state, config_dict)
        action = agent.select_action(norm_state)
        actions[i] = action
    return actions

#%% Training loop

if __name__ == "__main__":
    action_on_test_on = np.empty(100)
    action_on_test_off = np.empty(100)
    obs_dict = env.reset()
    
    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
    # TODO num_state = env.observation_space.n
    # TODO num_action = env.action_space.n
    agent = DQN(config_dict, opt, num_state=num_state) # num_action

    cumul_avg_reward = 0
    cumul_temp_offset = 0
    cumul_temp_error = 0
    cumul_signal_offset = 0
    cumul_signal_error = 0
    
    epsilon = 1.0

    for t in range(opt.nb_time_steps):
        if render:
            renderer.render(obs_dict)
            
        # Taking action with epsilon-greedy strategy
        if random.random() < epsilon:
            action = {k: random.randint(0,1) for k in obs_dict.keys()}
        else:
            action = {k: agent.select_action(normStateDict(obs_dict[k], config_dict)) for k in obs_dict.keys()}
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        
        if render and t >= opt.render_after:
            renderer.render(next_obs_dict)

        # Store transition in replay buffer
        for k in obs_dict.keys():
            agent.store_transition(normStateDict(
                obs_dict[k], config_dict), action[k], rewards_dict[k], normStateDict(next_obs_dict[k], config_dict))
            
            # Calculate metrics
            cumul_temp_offset += (next_obs_dict[k]["house_temp"] -
                                  next_obs_dict[k]["house_target_temp"]) / env.nb_agents
            cumul_temp_error += np.abs(next_obs_dict[k]["house_temp"] -
                                       next_obs_dict[k]["house_target_temp"]) / env.nb_agents
            cumul_avg_reward += rewards_dict[k] / env.nb_agents
        
        # Calculate metrics (mean values)
        cumul_signal_offset += next_obs_dict['0_1']["reg_signal"] - \
            next_obs_dict['0_1']["cluster_hvac_power"]
        cumul_signal_error += np.abs(
            next_obs_dict['0_1']["reg_signal"] - next_obs_dict['0_1']["cluster_hvac_power"])
        #print(next_obs_dict['0_1']["reg_signal"] - next_obs_dict['0_1']["cluster_hvac_power"])

        obs_dict = next_obs_dict # Set next_state as current state
        agent.update() # update agent
        epsilon *= opt.nb_time_steps/(t/(opt.nb_time_steps/20) + opt.nb_time_steps) # decrease epsilon
        
        # New episode, reset environment
        if t % time_steps_per_episode == time_steps_per_episode - 1:
            print("New episode at time {}".format(t))
            obs_dict = env.reset()

        # Update target network parameters
        if t % time_steps_per_epoch == time_steps_per_epoch - 1:
            print("Updating agent at time {}".format(t))
            agent.update_params()

        # Log train statistics
        if t % time_steps_train_log == time_steps_train_log - 1:
            #print("Logging stats at time {}".format(t))

            mean_avg_return = cumul_avg_reward/time_steps_train_log
            mean_temp_offset = cumul_temp_offset/time_steps_train_log
            mean_temp_error = cumul_temp_error/time_steps_train_log
            mean_signal_offset = cumul_signal_offset/time_steps_train_log
            mean_signal_error = cumul_signal_error/time_steps_train_log

            if log_wandb:
                wandb_run.log({"Mean train return": mean_avg_return, "Mean temperature offset": mean_temp_offset, "Mean temperature error": mean_temp_error,
                              "Mean signal offset": mean_signal_offset, "Mean signal error": mean_signal_error, "Training steps": t})

            # Reset variables
            cumul_temp_offset = 0
            cumul_temp_error = 0
            cumul_avg_reward = 0
            cumul_signal_offset = 0
            cumul_signal_error = 0

        # Test policy
        if t % time_steps_test_log == time_steps_test_log - 1:
            print("Testing at time {}".format(t))
            action_on_test_on = np.vstack((action_on_test_on, get_agent_test(agent, obs_dict["0_1"], 10, 30, config_dict, obs_dict["0_1"]["hvac_cooling_capacity"]/obs_dict["0_1"]["hvac_COP"])))
            action_on_test_off = np.vstack((action_on_test_off, get_agent_test(agent, obs_dict["0_1"], 10, 30, config_dict, 0.0)))
            # random.seed(t)
            # test_env = MADemandResponseEnv(config_dict, test=True)
            test_env = deepcopy(env)
            mean_test_return = testAgent(agent, test_env, opt.nb_time_steps_test)
            if log_wandb:
                wandb_run.log(
                    {"Mean test return": mean_test_return, "Training steps": t})
            else:
                print(
                    "Training step - {} - Mean test return: {}".format(t, mean_test_return))

    if render:
        renderer.__del__(obs_dict)
        
    # Plot
    action_on_test_on = action_on_test_on[1:]
    action_on_test_off = action_on_test_off[1:]
    colorPlotTestAgentHouseTemp(action_on_test_on, action_on_test_off, 10, 30, time_steps_test_log, log_wandb)
    # plot_env_test(env)

    # Save agent
    if opt.save_actor_name:
        agent.save_agent()

#%% Test
# from plotting import plot_agent_test
# import torch

# obs_dict = env.reset()
# num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
# agent = DQN(config_dict, opt, num_state=num_state) # num_action
# agent.policy_net.load_state_dict(torch.load('actors/dqn/actor.pth'))

# #%%
# plot_agent_test(env, agent, config_dict, 3000)
# %%


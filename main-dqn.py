#%% Imports

from env import *
from agents import *
from config import config_dict
from cli import cli_train

import os

import random
import numpy as np
import matplotlib.pyplot as plt
from agents.dqn import DQN
from copy import deepcopy

from env.MA_DemandResponse import MADemandResponseEnv as env
from utils import (
    normStateDict,
    testAgentHouseTemperature,
    saveActorNetDict,
    adjust_config,
    render_and_wandb_init,
)

os.environ["WANDB_SILENT"] = "true"

#%% Initializing

opt = cli_train() # get arguments from cli
adjust_config(opt, config_dict)
opt.no_wandb = True
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

#%%
# Training loop
if __name__ == "__main__":
    
    obs_dict = env.reset()
    
    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
    # num_state = env.observation_space.n
    # num_action = env.action_space.n
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
    
    agent.save_agent()
    
    # Plot
    # plot_env_test(env)

    # Save agent
    # if opt.save_actor_name:
    #     path = os.path.join(".", "actors", opt.save_actor_name)
    #     saveActorNetDict(agent, path)

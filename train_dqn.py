#%% Imports

from agents.dqn import DQN
from config import config_dict
from cli import cli_train
from env.MA_DemandResponse import MADemandResponseEnv
from metrics import Metrics
from plotting import colorPlotTestAgentHouseTemp
from utils import normStateDict, adjust_config, render_and_wandb_init, get_agent_test, test_dqn_agent

import random
import numpy as np

#%% Functions

def decrease(epsilon, opt, t):
    epsilon *= opt.nb_time_steps/(t/(opt.nb_time_steps/20) + opt.nb_time_steps)
    return epsilon
    
def train_dqn(env, agent, opt, config_dict, render, log_wandb, wandb_run):
    # Initialize render, if applicable
    if render:
        from env.renderer import Renderer
        renderer = Renderer(env.nb_agents)
    
    # Variables
    time_steps_per_episode = int(opt.nb_time_steps/opt.nb_tr_episodes)
    time_steps_per_epoch = int(opt.nb_time_steps/opt.nb_tr_epochs)
    time_steps_train_log = int(opt.nb_time_steps/opt.nb_tr_logs)
    time_steps_test_log = int(opt.nb_time_steps/opt.nb_test_logs)
    action_on_test_on = np.empty(100)
    action_on_test_off = np.empty(100)
    metrics = Metrics()
    epsilon = 1.0
        
    # Get first observation
    obs_dict = env.reset()

    for t in range(opt.nb_time_steps):
        
        # Render observation
        if render:
            renderer.render(obs_dict)
            
        # Select action with epsilon-greedy strategy
        if random.random() < epsilon:
            action = {k: random.randint(0,1) for k in obs_dict.keys()}
        else:
            action = {k: agent.select_action(normStateDict(obs_dict[k], config_dict)) for k in obs_dict.keys()}
        
        # Take action and get new transition
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        
        # Render next observation
        if render and t >= opt.render_after:
            renderer.render(next_obs_dict)

        # Store transition in replay buffer
        for k in obs_dict.keys():
            agent.store_transition(normStateDict(obs_dict[k], config_dict), action[k], rewards_dict[k], normStateDict(next_obs_dict[k], config_dict))
            
        # Update metrics
        metrics.update("0_1", next_obs_dict, rewards_dict, env)
        
        # Set next_state as current state
        obs_dict = next_obs_dict
        
        agent.update() # update agent
        epsilon = decrease(epsilon, opt, t)
        
        # New episode, reset environment
        if t % time_steps_per_episode == time_steps_per_episode - 1:
            print(f"New episode at time {t+1}")
            obs_dict = env.reset()

        # Update target network parameters
        if t % time_steps_per_epoch == time_steps_per_epoch - 1:
            print(f"Updating agent at time {t+1}")
            agent.update_params()

        # Log train statistics
        if t % time_steps_train_log == time_steps_train_log - 1:
            print(f"Logging stats at time {t+1}")
            logged_metrics = metrics.log(t, time_steps_train_log)
            if log_wandb:
                wandb_run.log(logged_metrics)
            metrics.reset()

        # Test policy
        if t % time_steps_test_log == time_steps_test_log - 1:
            print(f"Testing at time {t+1}")
            action_on_test_on = np.vstack((action_on_test_on, get_agent_test(agent, obs_dict["0_1"], config_dict, obs_dict["0_1"]["hvac_cooling_capacity"]/obs_dict["0_1"]["hvac_COP"])))
            action_on_test_off = np.vstack((action_on_test_off, get_agent_test(agent, obs_dict["0_1"], config_dict, 0.0)))
            mean_test_return = test_dqn_agent(agent, env, opt.nb_time_steps_test, config_dict, time_steps_per_episode)
            if log_wandb:
                wandb_run.log({"Mean test return": mean_test_return, "Training steps": t})
            else:
                print(f"Training step - {t+1} - Mean test return: {mean_test_return}")

    if render:
        renderer.__del__(obs_dict)
        
    # Plot
    colorPlotTestAgentHouseTemp(action_on_test_on, action_on_test_off, 10, 30, time_steps_test_log, log_wandb)

    # Save agent
    if opt.save_actor_name:
        agent.save()

#%% Train

if __name__ == "__main__":
    import os
    os.environ["WANDB_SILENT"] = "true"
    opt = cli_train()
    adjust_config(opt, config_dict)
    render, log_wandb, wandb_run = render_and_wandb_init(opt, config_dict)
    random.seed(opt.env_seed)
    env = MADemandResponseEnv(config_dict)
    agent = DQN(config_dict, opt)
    train_dqn(env, agent, opt, config_dict, render, log_wandb, wandb_run)
    
#%% Test
# from plotting import plot_agent_test
# import torch
# agent.policy_net.load_state_dict(torch.load('actors/dqn/actor.pth'))
# plot_agent_test(env, agent, config_dict, 3000)

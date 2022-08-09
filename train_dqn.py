#%% Imports

from agents.dqn import DQN
from config import config_dict
from cli import cli_train
from env.MA_DemandResponse import MADemandResponseEnv
from metrics import Metrics
from utils import adjust_config_train, normStateDict, render_and_wandb_init, test_dqn_agent, saveDQNNetDict

import random
import os
import numpy as np
import wandb

#%% Functions

def decrease(epsilon, opt):
    epsilon *= config_dict["DQN_prop"]["epsilon_decay"]
    epsilon = np.maximum(epsilon, config_dict["DQN_prop"]["min_epsilon"])
    return epsilon
    
def train_dqn(env, agent, opt, config_dict, render, log_wandb, wandb_run):
    id_rng = np.random.default_rng()
    unique_ID = str(int(id_rng.random() * 1000000))


    # Initialize render, if applicable
    if render:
        from env.renderer import Renderer
        renderer = Renderer(env.nb_agents)
    
    # Variables
    time_steps_per_episode = int(opt.nb_time_steps/opt.nb_tr_episodes)
    time_steps_train_log = int(opt.nb_time_steps/opt.nb_tr_logs)
    time_steps_test_log = int(opt.nb_time_steps/opt.nb_test_logs)
    time_steps_per_saving_actor = int(opt.nb_time_steps/(opt.nb_inter_saving_actor+1))

    metrics = Metrics()
    epsilon = 1.0
        
    # Get first observation
    obs_dict = env.reset()

    for t in range(opt.nb_time_steps):
        
        # Render observation
        if render:
            renderer.render(obs_dict)
            
        # Select action with epsilon-greedy strategy
        action = {}
        for k in obs_dict.keys():
            if random.random() < epsilon:
                action[k] = random.randint(0,1)
            else:
                action[k] = agent.select_action(normStateDict(obs_dict[k], config_dict))

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
            metrics.update(k, obs_dict, next_obs_dict, rewards_dict, env)
        
        # Set next_state as current state
        obs_dict = next_obs_dict
        
        agent.update() # update policy network

        epsilon = decrease(epsilon, config_dict)
        
        # New episode, reset environment
        if t % time_steps_per_episode == time_steps_per_episode - 1:
            print(f"New episode at time {t}")
            obs_dict = env.reset()

        # Log train statistics
        if t % time_steps_train_log == time_steps_train_log - 1:
            print(f"Logging stats at time {t}")
            logged_metrics = metrics.log(t, time_steps_train_log)
            if log_wandb:
                wandb_run.log(logged_metrics)
            metrics.reset()

        # Test policy
        if t % time_steps_test_log == time_steps_test_log - 1:
            print(f"Testing at time {t}")
            metrics_test = test_dqn_agent(agent, env, config_dict, opt, t)
            if log_wandb:
                wandb_run.log(metrics_test)
            else:
                print("Training step - {} - Mean test return: {}".format(t, metrics_test["Mean test return"]))


        if opt.save_actor_name and t % time_steps_per_saving_actor == 0 and t != 0:
            path = os.path.join(".", "actors", opt.save_actor_name + unique_ID)
            saveDQNNetDict(agent, path, t)
            if log_wandb:
                wandb.save(os.path.join(path, "DQN" + str(t) + ".pth"))

    if render:
        renderer.__del__(obs_dict)
        
    # Save agent
    if opt.save_actor_name:
        path = os.path.join(".", "actors", opt.save_actor_name + unique_ID)
        saveDQNNetDict(agent, path)
        if log_wandb:
            wandb.save(os.path.join(path, "DQN.pth"))
#%% Train

if __name__ == "__main__":
    import os
    os.environ["WANDB_SILENT"] = "true"
    opt = cli_train()
    adjust_config_train(opt, config_dict)
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

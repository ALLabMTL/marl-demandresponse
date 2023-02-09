#%% Imports

from copy import deepcopy
from collections import namedtuple
from logging import Logger

import numpy as np
import torch
from core.environment.environment import Environment
from utils.metrics import Metrics
from utils.utils import normStateDict
from core.agents.ppo import PPO
from config import config_dict
from utils.logger import logger

#%% Functions

ppo_props = {
    "PPO_prop": {
        "actor_layers": [100, 100],
        "critic_layers": [100, 100],
        "gamma": 0.99,
        "lr_critic": 3e-3,
        "lr_actor": 1e-3,
        "clip_param": 0.2,
        "max_grad_norm": 0.5,
        "ppo_update_time": 10,
        "batch_size": 256,
        "zero_eoepisode_return": False,
    },
}

def test_ppo_agent(agent, env: Environment, tr_time_steps):
    """
    Test ppo agent on an episode of nb_test_timesteps, with
    """
    nb_time_steps_test = 10
    env = deepcopy(env)
    cumul_avg_reward = 0
    cumul_temp_error = 0
    cumul_signal_error = 0
    obs_dict = env._reset()
    nb_agents = len(env.cluster.buildings)
    with torch.no_grad():
        for t in range(nb_time_steps_test):
            action_and_prob = {
                k: agent.select_action(normStateDict(obs_dict[k], config_dict))
                for k in obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
            obs_dict, rewards_dict = env._step(action)
            for i in range(nb_agents):
                cumul_avg_reward += rewards_dict[i] / nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["indoor_temp"] - obs_dict[i]["target_temp"])
                    / nb_agents
                )
                cumul_signal_error += np.abs(
                    obs_dict[i]["reg_signal"] - obs_dict[i]["cluster_hvac_power"]
                ) / (nb_agents**2)
    mean_avg_return = cumul_avg_reward / nb_time_steps_test
    mean_temp_error = cumul_temp_error / nb_time_steps_test
    mean_signal_error = cumul_signal_error / nb_time_steps_test

    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean signal error": mean_signal_error,
        "Training steps": tr_time_steps,
    }


def train_ppo(env: Environment, agent: PPO):

    nb_time_steps = 1000
    nb_test_logs = 100
    nb_tr_logs = 100
    nb_tr_epochs = 20
    # Initialize variables
    Transition = namedtuple(
        "Transition", ["state", "action", "a_log_prob", "reward", "next_state", "done"]
    )
    time_steps_per_episode = int(nb_time_steps / nb_tr_epochs)
    time_steps_per_epoch = int(nb_time_steps / nb_tr_epochs)
    time_steps_train_log = int(nb_time_steps / nb_tr_logs)
    time_steps_test_log = int(nb_time_steps / nb_test_logs)
    metrics = Metrics()

    # Get first observation
    obs_dict = env._reset()

    for t in range(nb_time_steps):


        # Select action with probabilities
        action_and_prob = {
            k: agent.select_action(normStateDict(obs_dict[k], config_dict))
            for k in obs_dict.keys()
        }
        action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
        action_prob = {k: action_and_prob[k][1] for k in obs_dict.keys()}

        # Take action and get new transition
        next_obs_dict, rewards_dict = env._step(action)

        # Episode is done
        done = t % time_steps_per_episode == time_steps_per_episode - 1
        logger.debug(obs_dict.keys())
        # Storing in replay buffer
        for k in obs_dict.keys():
            agent.store_transition(
                Transition(
                    normStateDict(obs_dict[k], config_dict),
                    action[k],
                    action_prob[k],
                    rewards_dict[k],
                    normStateDict(next_obs_dict[k], config_dict),
                    done,
                ),
                k,
            )
            # Update metrics
            metrics.update(k, obs_dict, next_obs_dict, rewards_dict, env)

        # Set next state as current state
        obs_dict = next_obs_dict

        # New episode, reset environment
        if done:
            logger.info(f"New episode at time {t}")
            obs_dict = env._reset()

        # Epoch: update agent
        if (
            t % time_steps_per_epoch == time_steps_per_epoch - 1
            and len(agent.buffer[0]) >= agent.batch_size
        ):
            logger.info(f"Updating agent at time {t}")
            agent.update(t)

        # Log train statistics
        if t % time_steps_train_log == time_steps_train_log - 1:  # Log train statistics
            logger.info("Logging stats at time {}".format(t))
            logged_metrics = metrics.log(t, time_steps_train_log)
            logger.info(f"Stats : {logged_metrics}")

            metrics.reset()

        # Test policy
        if t % time_steps_test_log == time_steps_test_log - 1:  # Test policy
            logger.info(f"Testing at time {t}")
            metrics_test = test_ppo_agent(agent, env, t)
            logger.info(f"Metrics test: {metrics_test}")
            logger.info("Training step - {}".format(t))
        
    logger.info("Simulation ended")
    raise SystemExit()
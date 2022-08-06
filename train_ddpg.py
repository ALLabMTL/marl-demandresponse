#%% Imports

from config import config_dict
from cli import cli_train
from agents.ddpg import MADDPG
from env.MA_DemandResponse import MADemandResponseEnv
from metrics import Metrics

from plotting import colorPlotTestAgentHouseTemp
from utils import (
    normStateDict,
    testAgentHouseTemperature,
    saveActorNetDict,
    adjust_config_train,
    render_and_wandb_init,
    test_ppo_agent,
)
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from collections import namedtuple
import wandb

#%% Functions

def get_dim_info(opt, env, n_state, n_action=2):
    """get the dimension information of the environment"""
    dim_info = {}
    for agent_id in range(opt.nb_agents):
        dim_info[agent_id] = []
        dim_info[agent_id].append(n_state)
        dim_info[agent_id].append(n_action)
    return dim_info

def train_ddpg(env, dim_info, config_dict, opt):
    # id_rng = np.random.default_rng()
    # unique_ID = str(int(id_rng.random() * 1000000))
    maddpg = MADDPG(
        dim_info,
        config_dict,
        opt
    )

    step = 0  # global step counter
    # agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(opt.episode_num) for agent_id in range(opt.nb_agents)}
    for episode in range(opt.episode_num):
        obs = env.reset()
        obs = normStateDict(obs[next(iter(obs))], config_dict)
        obs_ = {
            agent_id: obs # env.action_space(agent_id).sample()
            for agent_id in range(opt.nb_agents)
        }
        agent_reward = {
            agent_id: 0 for agent_id in range(opt.nb_agents)
        }  # agent reward of the current episode
        for s in range(opt.episode_length):  # interact with the env for an episode
            step += 1
            if step < opt.random_steps:
                action = {
                    agent_id: np.random.randint(0,2) # env.action_space(agent_id).sample()
                    for agent_id in range(opt.nb_agents)
                }
            else:
                action = maddpg.select_action(obs_)

            next_obs, reward, done, info = env.step(action)
            # env.render()
            next_obs = normStateDict(next_obs[next(iter(next_obs))], config_dict)
            next_obs_ = {
                agent_id: next_obs # env.action_space(agent_id).sample()
                for agent_id in range(opt.nb_agents)
            }
            maddpg.push(obs_, action, reward, next_obs_, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if (
                step >= opt.random_steps and step % opt.learn_interval == 0
            ):  # learn every few steps
                # maddpg.update(opt.batch_size, opt.gamma)
                maddpg.update()
                maddpg.update_target()

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f"episode {episode + 1}, "
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f"{agent_id}: {r:>4f}; "
                sum_reward += r
            message += f"sum reward: {sum_reward}"
            print(message)

    maddpg.save(episode_rewards)  # save model
    return episode_rewards


def get_running_reward(arr: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(arr)
    for i in range(window - 1):
        running_reward[i] = np.mean(arr[: i + 1])
    for i in range(window - 1, len(arr)):
        running_reward[i] = np.mean(arr[i - window + 1 : i + 1])
    return running_reward


#%% Train

if __name__ == "__main__":
    # import os

    os.environ["WANDB_SILENT"] = "true"
    opt = cli_train()
    adjust_config_train(opt, config_dict)
    # render, log_wandb, wandb_run = render_and_wandb_init(opt, config_dict)
    random.seed(opt.env_seed)
    # env = MADemandResponseEnv(config_dict)
    # agent = PPO(config_dict, opt)
    # train_ppo(env, agent, opt, config_dict, render, log_wandb, wandb_run)
    from easydict import EasyDict

    opt = EasyDict(vars(opt))
    opt.env_name = "MA_DemandResponse"
    opt.episode_num = 10000
    opt.episode_length = 25
    opt.random_steps = 100
    opt.soft_tau = 0.02
    opt.gamma = 0.95
    opt.buffer_capacity = int(1e6)
    opt.batch_size = 64
    opt.actor_lr = 1e-2
    opt.critic_lr = 1e-2
    opt.learn_interval = 100

    env_dir = os.path.join("./ddpg_results")
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f"{total_files + 1}")
    os.makedirs(result_dir)
    opt.result_dir = result_dir

    env = MADemandResponseEnv(config_dict)
    obs_dict = env.reset()
    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
    dim_info = get_dim_info(opt, env, num_state)
    episode_rewards = train_ddpg(env, dim_info, config_dict, opt)

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, opt.episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    title = f"training result of maddpg solve {opt.env_name}"
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from app.core.agents.agent import Agent
from app.core.agents.buffer import DDPGBuffer as Buffer
from app.core.agents.ddpg import DDPG, DDPGProperties, get_dim_info


class MADDPG(DDPG, Agent):
    def __init__(self, static_props: DDPGProperties, opt, num_state, wandb_run):
        # sum all the dims of each agent to get input dim for critic
        # global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        self.config_dict = static_props
        self.static_props = static_props
        self.opt = opt
        self.num_state = num_state
        self.capacity = self.static_props.buffer_capacity
        dim_info = get_dim_info(opt, num_state)
        global_state_action_dim = sum(sum(val) for val in dim_info.values())
        for agent_id, (state_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = DDPG(
                self.config_dict,
                self.opt,
                global_state_action_dim,
                state_dim,
                action_dim,
            )
            self.buffers[agent_id] = Buffer(
                self.capacity,
                state_dim,
                action_dim,
                "cpu",
            )
        self.shared = opt.DDPG_shared
        assert self.shared != -1, "shared must be set as True or False (1 or 0)"
        print("DDPG shared status: {}".format(self.shared))
        if self.shared:
            self.agents[agent_id].actor_net = self.agents[0].actor_net
            self.agents[agent_id].critic_net = self.agents[0].critic_net
            self.agents[agent_id].tgt_actor_net = self.agents[0].tgt_actor_net
            self.agents[agent_id].tgt_critic_net = self.agents[0].tgt_critic_net
            self.agents[agent_id].actor_optimizer = self.agents[0].actor_optimizer
            self.agents[agent_id].critic_optimizer = self.agents[0].critic_optimizer

        self.dim_info = dim_info
        self.wandb_run = wandb_run
        self.batch_size = self.static_props.batch_size
        self.result_dir = os.path.join(
            "./ddpg_results"
        )  # directory to save the training result
        self.logger = self.setup_logger(os.path.join(self.result_dir, "maddpg.log"))
        self.log_wandb = not opt.no_wandb

    def setup_logger(self, filename):
        """set up logger with filename."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(filename, mode="w")
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s--%(levelname)s--%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def push(self, state, action, reward, next_state, done):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in state.keys():
            s = state[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]

            r = reward[agent_id]
            next_s = next_state[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].push(s, a, r, next_s, d)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        # obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        state, action, reward, next_state, done, next_action = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            s, a, r, n_s, d = buffer.sample(indices)
            state[agent_id] = s
            action[agent_id] = a
            reward[agent_id] = r
            next_state[agent_id] = n_s
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_action[agent_id] = self.agents[agent_id].select_action(
                n_s, is_target=True
            )

        # return obs, act, reward, next_obs, done, next_act
        return state, action, reward, next_state, done, next_action

    def select_action(self, state):
        # select action for each agent
        actions = {}
        for agent, s in state.items():
            s = torch.from_numpy(s).unsqueeze(0).float()
            a = self.agents[agent].select_action(s)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            self.logger.info(f"{agent} action: {actions[agent]}")
        return actions

    def update_target(self):
        # update target network for all the agents
        for agent in self.agents.values():
            agent.update_target()

    def update(self):
        for agent_id, agent in self.agents.items():
            state, action, reward, next_state, done, next_action = self.sample(
                self.batch_size
            )

            # update critic
            critic_value = agent.get_value(
                list(state.values()), list(action.values()), is_target=False
            )

            # calculate target critic value
            next_target_critic_value = agent.get_value(
                list(next_state.values()), list(next_action.values()), is_target=True
            )
            target_value = reward[agent_id] + self.agents[
                agent_id
            ].gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(
                critic_value, target_value.detach(), reduction="mean"
            )
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action_, logits = agent.select_action(state[agent_id], output_logits=True)
            # print("action", action)
            # print("agent_id", agent_id)
            action[agent_id] = action_
            actor_loss = -agent.get_value(
                list(state.values()), list(action.values())
            ).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
        if self.log_wandb:
            self.wandb_run.log({"critic_loss": critic_loss, "actor_loss": actor_loss})

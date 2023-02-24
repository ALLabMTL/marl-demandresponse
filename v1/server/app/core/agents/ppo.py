import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from app.core.agents.network import Actor, Critic
from app.utils.logger import logger


class PPO:
    def __init__(self, config_dict: dict, num_state=22, num_action=2, seed=1):
        super(PPO, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # if True:
        #    self.actor_net = OldActor(num_state=num_state, num_action=num_action)
        #    self.critic_net = OldCritic(num_state=num_state)
        self.actor_net = Actor(
            num_state=num_state,
            num_action=num_action,
            layers=config_dict["PPO_prop"]["actor_layers"],
        ).to(self.device)
        self.critic_net = Critic(
            num_state=num_state, layers=config_dict["PPO_prop"]["critic_layers"]
        ).to(self.device)
        # TODO: change this static value (only for tests)
        self.nb_agents = 10
        self.batch_size = config_dict["PPO_prop"]["batch_size"]
        self.ppo_update_time = config_dict["PPO_prop"]["ppo_update_time"]
        self.max_grad_norm = config_dict["PPO_prop"]["max_grad_norm"]
        self.clip_param = config_dict["PPO_prop"]["clip_param"]
        self.gamma = config_dict["PPO_prop"]["gamma"]
        self.lr_actor = config_dict["PPO_prop"]["lr_actor"]
        self.lr_critic = config_dict["PPO_prop"]["lr_critic"]
        self.zero_eoepisode_return = config_dict["PPO_prop"]["zero_eoepisode_return"]

        # Initialize buffer
        self.buffer = {}
        for agent in range(self.nb_agents):
            self.buffer[agent] = []

        logger.info(
            "ppo_update_time: {}, max_grad_norm: {}, clip_param: {}, gamma: {}, batch_size: {}, lr_actor: {}, lr_critic: {}".format(
                self.ppo_update_time,
                self.max_grad_norm,
                self.clip_param,
                self.gamma,
                self.batch_size,
                self.lr_actor,
                self.lr_critic,
            )
        )
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), self.lr_critic
        )

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob.cpu())
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        # state = torch.from_numpy(state)
        state = state.to(self.device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.cpu().item()

    def reset_buffer(self):
        self.buffer = {}
        for agent in range(self.nb_agents):
            self.buffer[agent] = []

    def store_transition(self, transition, agent):
        self.buffer[agent].append(transition)

    def update(self, t):
        sequential_buffer = []
        for agent in range(self.nb_agents):
            sequential_buffer += self.buffer[agent]

        state_np = np.array([t.state for t in sequential_buffer])
        next_state_np = np.array([t.next_state for t in sequential_buffer])
        action_np = np.array([t.action for t in sequential_buffer])
        old_action_log_prob_np = np.array([t.a_log_prob for t in sequential_buffer])

        state = torch.tensor(state_np, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state_np, dtype=torch.float).to(self.device)
        action = torch.tensor(action_np, dtype=torch.long).view(-1, 1).to(self.device)
        reward = [t.reward for t in sequential_buffer]
        old_action_log_prob = (
            torch.tensor(old_action_log_prob_np, dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        done = [t.done for t in sequential_buffer]

        """
        # Changed to accelerate process. UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray $

        state = torch.tensor([t.state for t in sequential_buffer], dtype=torch.float) 
        next_state = torch.tensor([t.next_state for t in sequential_buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in sequential_buffer], dtype=torch.long).view(-1, 1) 
        reward = [t.reward for t in sequential_buffer] 

        old_action_log_prob = torch.tensor( 
            [t.a_log_prob for t in sequential_buffer], dtype=torch.float 
        ).view(-1, 1) 
        done = [t.done for t in sequential_buffer] 
        
        """

        Gt = []
        for i in reversed(range(len(reward))):
            if done[i]:
                if self.zero_eoepisode_return:
                    R = 0
                else:
                    R = self.get_value(
                        next_state[i]
                    )  # When last state of episode, start from estimated value of next state
            R = reward[i] + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(self.device)
        ratios = np.array([])
        clipped_ratios = np.array([])
        gradient_norms = np.array([])
        logger.info("The agent is updating....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(
                SubsetRandomSampler(range(len(sequential_buffer))),
                self.batch_size,
                False,
            ):
                if self.training_step % 1000 == 0:
                    logger.info(
                        "Time step: {} ，train {} times".format(t, self.training_step)
                    )
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)

                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core
                action_prob = self.actor_net(state[index]).gather(
                    1, action[index]
                )  # new policy
                ratio = action_prob / old_action_log_prob[index]
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_param, 1 + self.clip_param
                )
                ratios = np.append(ratios, ratio.cpu().detach().numpy())
                clipped_ratios = np.append(
                    clipped_ratios, clipped_ratio.cpu().detach().numpy()
                )

                surr1 = ratio * advantage
                surr2 = clipped_ratio * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm
                )
                gradient_norms = np.append(gradient_norms, gradient_norm.cpu().detach())
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm
                )
                self.critic_net_optimizer.step()
                self.training_step += 1

        self.reset_buffer()  # clear experience

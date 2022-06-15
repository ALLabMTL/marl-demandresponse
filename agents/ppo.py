import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import time
import wandb
from agents.network import Actor, Critic, OldActor, OldCritic


class PPO():
    def __init__(self, config_dict, opt, gamma=0.99, buffer_capacity=524288, ppo_update_time=10, max_grad_norm=0.5, clip_param=0.2, num_state=22, num_action=2):
        super(PPO, self).__init__()
        self.seed = opt.net_seed
        torch.manual_seed(self.seed)

        self.batch_size = opt.batch_size

        if True:
            self.actor_net = OldActor(num_state=num_state, num_action=num_action)
            self.critic_net = OldCritic(num_state=num_state)            
        #self.actor_net = Actor(num_state=num_state, num_action=num_action, layers=config_dict["nn_prop"]["actor_layers"])
        #self.critic_net = Critic(num_state=num_state, layers=config_dict["nn_prop"]["critic_layers"])
        self.buffer = []
        self.buffer_capacity = buffer_capacity
        self.ppo_update_time = ppo_update_time
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.gamma = gamma
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)

    def select_action(self, state, temp=1):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state, temp)
        #print(action_prob)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()
    
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, t):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('Time step: {} ，train {} times'.format(t, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)

                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                
                # epoch iteration, PPO core
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                #self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience
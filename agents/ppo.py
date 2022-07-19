import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import time
import wandb
import numpy as np
from agents.network import Actor, Critic, OldActor, OldCritic


class PPO():
    def __init__(self, config_dict, opt, num_state=22, num_action=2, wandb_run = None):
        super(PPO, self).__init__()
        self.seed = opt.net_seed
        torch.manual_seed(self.seed)


        #if True:
        #    self.actor_net = OldActor(num_state=num_state, num_action=num_action)
        #    self.critic_net = OldCritic(num_state=num_state)            
        self.actor_net = Actor(num_state=num_state, num_action=num_action, layers=config_dict["PPO_prop"]["actor_layers"])
        self.critic_net = Critic(num_state=num_state, layers=config_dict["PPO_prop"]["critic_layers"])
        self.buffer = []
        self.batch_size = config_dict["PPO_prop"]["batch_size"]
        self.ppo_update_time = config_dict["PPO_prop"]["ppo_update_time"]
        self.max_grad_norm = config_dict["PPO_prop"]["max_grad_norm"]
        self.clip_param = config_dict["PPO_prop"]["clip_param"]
        self.gamma = config_dict["PPO_prop"]["gamma"]
        self.lr_actor = config_dict["PPO_prop"]["lr_actor"]
        self.lr_critic = config_dict["PPO_prop"]["lr_critic"]
        self.wandb_run = wandb_run
        self.log_wandb = not opt.no_wandb

        print("ppo_update_time: {}, max_grad_norm: {}, clip_param: {}, gamma: {}, batch_size: {}, lr_actor: {}, lr_critic: {}".format(
            self.ppo_update_time, self. max_grad_norm, self.clip_param, self.gamma, self.batch_size, self.lr_actor, self.lr_critic))
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), self.lr_critic)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
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

        ratios = np.array([])
        clipped_ratios = np.array([])
        gradient_norms = np.array([])
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
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) 


                ratios = np.append(ratios, ratio.detach().numpy())
                clipped_ratios = np.append(clipped_ratios, clipped_ratio.detach().numpy())

                surr1 = ratio * advantage
                surr2 = clipped_ratio * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                gradient_norms = np.append(gradient_norms, gradient_norm.detach())
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                #self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        if self.log_wandb:

            max_ratio = np.max(ratios)
            mean_ratio = np.mean(ratios)
            median_ratio = np.median(ratios)
            min_ratio = np.min(ratios)
            per95_ratio = np.percentile(ratios, 95)
            per75_ratio = np.percentile(ratios, 75)
            per25_ratio = np.percentile(ratios, 25)
            per5_ratio = np.percentile(ratios, 5)
            max_cl_ratio = np.max(clipped_ratios)
            mean_cl_ratio = np.mean(clipped_ratios)
            median_cl_ratio = np.median(clipped_ratios)
            min_cl_ratio = np.min(clipped_ratios)
            per95_cl_ratio = np.percentile(clipped_ratios, 95)
            per75_cl_ratio = np.percentile(clipped_ratios, 75)
            per25_cl_ratio = np.percentile(clipped_ratios, 25)
            per5_cl_ratio = np.percentile(clipped_ratios, 5)
            max_gradient_norm = np.max(gradient_norms)
            mean_gradient_norm = np.mean(gradient_norms)
            median_gradient_norm = np.median(gradient_norms)
            min_gradient_norm = np.min(gradient_norms)
            per95_gradient_norm = np.percentile(gradient_norms, 95)
            per75_gradient_norm = np.percentile(gradient_norms, 75)
            per25_gradient_norm = np.percentile(gradient_norms, 25)
            per5_gradient_norm = np.percentile(gradient_norms, 5)

            self.wandb_run.log({
                "PPO max ratio": max_ratio,
                "PPO mean ratio": mean_ratio,
                "PPO median ratio": median_ratio,
                "PPO min ratio": min_ratio,
                "PPO ratio 95 percentile": per95_ratio,
                "PPO ratio 5 percentile": per5_ratio,
                "PPO ratio 75 percentile": per75_ratio,
                "PPO ratio 25 percentile": per25_ratio,
                "PPO max clipped ratio": max_cl_ratio,
                "PPO mean clipped ratio": mean_cl_ratio,
                "PPO median clipped ratio": median_cl_ratio,
                "PPO min clipped ratio": min_cl_ratio,
                "PPO clipped ratio 95 percentile": per95_cl_ratio,
                "PPO clipped ratio 5 percentile": per5_cl_ratio,
                "PPO clipped ratio 75 percentile": per75_cl_ratio,
                "PPO clipped ratio 25 percentile": per25_cl_ratio,
                "PPO max gradient norm": max_gradient_norm,
                "PPO mean gradient norm": mean_gradient_norm,
                "PPO median gradient norm": median_gradient_norm,
                "PPO min gradient norm": min_gradient_norm,
                "PPO gradient norm 95 percentile": per95_gradient_norm,
                "PPO gradient norm 5 percentile": per5_gradient_norm,
                "PPO gradient norm 75 percentile": per75_gradient_norm,
                "PPO gradient norm 25 percentile": per25_gradient_norm,
                "Training steps": t})

        del self.buffer[:] # clear experience
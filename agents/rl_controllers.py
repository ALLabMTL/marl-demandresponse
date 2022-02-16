import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from agents.network import Actor
import sys
sys.path.append("..")
from utils import normStateDict


class PPOAgent():
    def __init__(self, agent_properties, num_state=18, num_action=2):
        super(PPOAgent, self).__init__()
        self.id = agent_properties["id"]
        self.actor_name = agent_properties["actor_name"]
        self.actor_path = os.path.join(".","actors",self.actor_name)

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)


        self.actor_net = Actor(num_state=num_state, num_action=num_action)
        self.actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'actor.pth')))
        self.actor_net.eval()

        self.temp = agent_properties["exploration_temp"]


    def act(self, obs_dict):
        state = normStateDict(obs_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state, self.temp)
        #print(action_prob)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

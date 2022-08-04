from utils import normStateDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from agents.network import Actor, OldActor
import sys
sys.path.append("..")


class PPOAgent():
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        super(PPOAgent, self).__init__()
        self.id = agent_properties["id"]
        self.actor_name = agent_properties["actor_name"]
        self.actor_path = os.path.join(".", "actors", self.actor_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.actor_net = Actor(num_state=num_state, num_action=num_action, layers = config_dict["PPO_prop"]["actor_layers"])
        self.actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'actor.pth')))
        self.actor_net.eval()


    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item()

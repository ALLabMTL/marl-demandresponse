import os

import torch
from torch.distributions import Categorical

from app.core.agents.controllers.controller import Controller
from v0.agents.network import Actor, DDPG_Network, DQN_network
from v0.utils import normStateDict


class PPOController(Controller):
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        self.id = agent_properties["id"]
        self.actor_name = agent_properties["actor_name"]
        self.actor_path = os.path.join(".", "actors", self.actor_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.actor_net = Actor(
            num_state=num_state,
            num_action=num_action,
            layers=config_dict["PPO_prop"]["actor_layers"],
        )
        self.actor_net.load_state_dict(
            torch.load(
                os.path.join(self.actor_path, "actor.pth"),
                map_location=torch.device("cpu"),
            )
        )
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


class DQNController(Controller):
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        self.id = agent_properties["id"]
        self.agent_name = agent_properties["actor_name"]
        self.agent_path = os.path.join(".", "actors", self.agent_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.DQN_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=config_dict["DQN_prop"]["network_layers"],
        )
        self.DQN_net.load_state_dict(
            torch.load(os.path.join(self.agent_path, "DQN.pth"))
        )
        self.DQN_net.eval()

    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.DQN_net(state)
        action = torch.argmax(qs).item()
        return action


class DDPGController(Controller):
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        self.id = agent_properties["id"]
        self.agent_name = agent_properties["actor_name"]
        self.agent_path = os.path.join(".", "actors", self.agent_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.DDPG_net = DDPG_Network(
            in_dim=num_state,
            out_dim=num_action,
            hidden_dim=config_dict["DDPG_prop"]["actor_hidden_dim"],
        )
        self.DDPG_net.load_state_dict(
            torch.load(os.path.join(self.agent_path, "DDPG.pth"))
        )
        self.DDPG_net.eval()

    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.DDPG_net(state)
        action = torch.argmax(qs).item()
        return action

import os

import torch
from torch.distributions import Categorical
from v0.agents.network import Actor, DDPG_Network, DQN_network
from v0.utils import normStateDict

from app.core.agents.controllers.controller import Controller

"""
Module: PPOController, DQNController, DDPGController

This module provides classes that act as controllers for reinforcement learning agents that use PPO, DQN or DDPG algorithms respectively.

Classes:
- PPOController: Implements a controller for agents that use PPO algorithm for training. It loads the actor network model from a saved file and uses it to make decisions about the action to take in response to observations.
- DQNController: Implements a controller for agents that use DQN algorithm for training. It loads the DQN network model from a saved file and uses it to make decisions about the action to take in response to observations.
- DDPGController: Implements a controller for agents that use DDPG algorithm for training. It loads the DDPG network model from a saved file and uses it to make decisions about the action to take in response to observations.

Methods:
- __init__(self, agent_properties, config_dict, num_state=22, num_action=2): Initializes the controller object with the given agent properties and configuration parameters. Loads the appropriate network model from a saved file and sets it up for inference.
- act(self, obs_dict): Takes in a dictionary of observation values and returns an action for the agent to perform based on the algorithm being used.

"""


class PPOController(Controller):
    """
    Implements a Proximal Policy Optimization (PPO) controller for an agent in a multi-agent system.

    Attributes:
        id (str): The id of the agent.
        actor_name (str): The name of the actor used for the PPO controller.
        actor_path (str): The path to the actor file for the PPO controller.
        config_dict (dict): A dictionary of configuration parameters for the controller.
        seed (int): The seed used for random number generation.
        actor_net (Actor): The actor network used for the PPO controller.

    Methods:
        act(obs_dict): Takes in a dictionary of observations from the environment and returns an action to take.
    """

    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        """
        Initializes a PPOController object.

        Paramters:
        - agent_properties: A dictionary containing information about the agent, including its ID,
          name, and seed.
        - config_dict: A dictionary containing configuration information for the agent.
        - num_state: An integer representing the number of states in the agent's environment.
        - num_action: An integer representing the number of possible actions in the agent's environment.

        Returns:
        None
        """
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
        """
        Takes an observation dictionary and returns an action.

        Parameter:
        - obs_dict: A dictionary containing observations for the agent's environment.

        Returns:
        An integer representing the chosen action.
        """
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item()


class DQNController(Controller):
    """
    Implements a Deep Q-Network (DQN) controller for an agent in a multi-agent system.

    Attributes:
        id (str): The id of the agent.
        agent_name (str): The name of the agent used for the DQN controller.
        agent_path (str): The path to the agent file for the DQN controller.
        config_dict (dict): A dictionary of configuration parameters for the controller.
        seed (int): The seed used for random number generation.
        DQN_net (DQN_network): The DQN network used for the DQN controller.

    Methods:
        act(obs_dict): Takes in a dictionary of observations from the environment and returns an action to take.
    """

    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        """
        Initializes a DQNController instance.

        Parameters:
        - agent_properties (dict): A dictionary containing information about the agent.
        - config_dict (dict): A dictionary containing configuration information.
        - num_state (int): The number of states in the environment.
        - num_action (int): The number of actions in the environment.

        Returns:
            None
        """
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
        """
        Returns the action selected by the DQN network given the current observation.

        Parameter:
        - obs_dict (dict): A dictionary containing the current observation.

        Returns:
            int: The index of the action selected by the network.
        """
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.DQN_net(state)
        action = torch.argmax(qs).item()
        return action


class DDPGController(Controller):
    """
    Implements a Deep Deterministic Policy Gradient (DDPG) controller for an agent in a multi-agent system.

    Attributes:
        id (str): The id of the agent.
        agent_name (str): The name of the agent used for the DDPG controller.
        agent_path (str): The path to the agent file for the DDPG controller.
        config_dict (dict): A dictionary of configuration parameters for the controller.
        seed (int): The seed used for random number generation.
        DDPG_net (DDPG_Network): The DDPG network used for the DDPG controller.

    """

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

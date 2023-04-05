import os
import random
from typing import Dict, List

import numpy as np
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim

from app.core.agents.trainables.buffer import ReplayBuffer, Transition
from app.core.agents.trainables.network import DQN_network
from app.core.agents.trainables.trainable import Trainable
from app.core.environment.environment_properties import EnvironmentObsDict


class DQNProperties(pydantic.BaseModel):
    """Properties for DQN agent."""

    network_layers: list[int] = pydantic.Field(
        default=[100, 100],
        description="List of layer sizes for the DQN network.",
    )
    gamma: float = pydantic.Field(
        default=0.99,
        description="Discount factor for the reward.",
    )
    tau: float = pydantic.Field(
        default=0.001,
        description="Soft target update parameter.",
    )
    lr: float = pydantic.Field(
        default=3e-3,
        description="Learning rate for the DQN network.",
    )
    buffer_capacity: int = pydantic.Field(
        default=524288,
        description="Capacity of the replay buffer.",
    )
    batch_size: int = pydantic.Field(
        default=256,
        description="Batch size for the DQN agent.",
    )
    epsilon_decay: float = pydantic.Field(
        default=0.99998,
        description="Epsilon decay rate for the DQN agent.",
    )
    min_epsilon: float = pydantic.Field(
        default=0.01,
        description="Minimum epsilon for the DQN agent.",
    )


class DQN(Trainable):
    def __init__(
        self, config: DQNProperties, num_state=22, num_action=2, seed=1
    ) -> None:
        self.seed = seed
        self.epsilon = 1.0
        self.last_actions: Dict[int, bool] = {}

        torch.manual_seed(self.seed)
        self.agent_prop = config
        self.inner_layers = config.network_layers
        self.gamma = config.gamma
        self.tau = config.tau
        self.buffer_cap = config.buffer_capacity
        self.lr = config.lr
        self.batch_size = config.batch_size

        self.policy_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=config.network_layers,
        )
        self.target_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=config.network_layers,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())  # same weights

        self.buffer = ReplayBuffer(self.buffer_cap)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if self.device.type == 'cuda':
        #    self.policy_net.to(self.device)
        #    self.target_net.to(self.device)

        # TODO weight decay?
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), self.lr)

    def select_actions(self, observations: List[np.ndarray]) -> Dict[int, bool]:
        # Select action with epsilon-greedy strategy
        actions: Dict[int, bool] = {}
        for observation_id, observation in enumerate(observations):
            if random.random() < self.epsilon:
                actions[observation_id] = bool(random.randint(0, 1))
            else:
                state = torch.from_numpy(observation).float().unsqueeze(0)
                with torch.no_grad():
                    qs = self.policy_net(state)
                    actions[observation_id] = bool(torch.argmax(qs).item())
        self.last_actions = actions
        return actions

    def store_transition(
        self,
        observations: Dict[int, EnvironmentObsDict],
        next_observations: Dict[int, EnvironmentObsDict],
        rewards: Dict[int, float],
        done: bool,
    ) -> None:
        for agent_id, observation in enumerate(observations):
            single_obs = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            single_action = torch.tensor(
                [[self.last_actions[agent_id]]], dtype=torch.long
            )
            single_rewards = torch.tensor([[rewards[agent_id]]], dtype=torch.float)
            single_next_obs = torch.tensor(
                next_observations[agent_id], dtype=torch.float
            ).unsqueeze(0)
            self.buffer.push(single_obs, single_action, single_rewards, single_next_obs)

    def convert_batch(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.cat(batch.state)
        a = torch.cat(batch.action)
        r = torch.cat(batch.reward)
        s_next = torch.cat(batch.next_state)
        # if self.device.type == 'cuda':
        #    s.to(self.device)
        #    a.to(self.device)
        #    r.to(self.device)
        #    s_next.to(self.device)
        return s, a, r, s_next

    def update_target_network(self) -> None:
        new_params = self.policy_net.state_dict()
        params = self.target_net.state_dict()
        for k in params.keys():
            # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
            params[k] = (1 - self.tau) * params[k] + self.tau * new_params[k]
        self.target_net.load_state_dict(params)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return  # exit if there are not enough transitions in buffer

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, next_state = self.convert_batch(batch)

        # Compute Q(s,a) from state
        q_values = self.policy_net(state).gather(1, action)

        # Compute max Q(s',a) from next state
        next_q_values = self.target_net(next_state).max(1)[0].detach()

        # Compute expected Q(s,a)
        expected_q_values = reward + (next_q_values.unsqueeze(1) * self.gamma)

        # Compute loss
        criterion = nn.SmoothL1Loss()  # Huber with beta/delta=1 (default)
        loss = criterion(q_values, expected_q_values)

        # Optimize the model
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # clamp gradients
        self.policy_optimizer.step()

        self.update_target_network()

        self.decrease_epsilon()

    def decrease_epsilon(self) -> None:
        new_epsilon = self.agent_prop.epsilon_decay
        self.epsilon = np.maximum(new_epsilon, self.agent_prop.min_epsilon)

    # TODO: Move this to abstract class
    def save(self, path, t=None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        actor_net = self.policy_net
        if t:
            torch.save(
                actor_net.state_dict(), os.path.join(path, "actor" + str(t) + ".pth")
            )
        else:
            torch.save(actor_net.state_dict(), os.path.join(path, "actor.pth"))


class DDQN(DQN):
    def __init__(
        self, config: DQNProperties, num_state=22, num_action=2, seed=1
    ) -> None:
        super().__init__(config, num_state, num_action, seed)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return  # exit if there are not enough transitions in buffer

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, next_state = self.convert_batch(batch)

        # Compute Q(s,a) from state
        q_values = self.policy_net(state).gather(1, action)
        next_action = self.policy_net(next_state).argmax(dim=1, keepdim=True)

        # Compute max Q(s',a) from next state
        next_q_values = self.target_net(next_state).gather(1, next_action).detach()

        # Compute expected Q(s,a)
        expected_q_values = reward + (next_q_values.unsqueeze(1) * self.gamma)

        # Compute loss
        criterion = nn.SmoothL1Loss()  # Huber with beta/delta=1 (default)
        loss = criterion(q_values, expected_q_values)

        # Optimize the model
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # clamp gradients
        self.policy_optimizer.step()


if __name__ == "__main__":
    pass

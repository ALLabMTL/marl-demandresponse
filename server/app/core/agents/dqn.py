import os
import random

import numpy as np
import pydantic
import torch
import torch.nn as nn
import torch.optim as optim

from app.core.agents.buffer import ReplayBuffer, Transition
from server.app.core.agents.trainables.network import DQN_network
from server.app.core.agents.trainables.trainable import Trainable


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
    def __init__(self, config_dict, num_state=22, num_action=2, seed=2) -> None:
        super().__init__()
        self.seed = seed
        self.epsilon = 1.0

        torch.manual_seed(self.seed)

        self.agent_prop = config_dict["DQN_prop"]
        self.inner_layers = self.agent_prop["network_layers"]
        self.gamma = self.agent_prop["gamma"]
        self.tau = self.agent_prop["tau"]
        self.buffer_cap = self.agent_prop["buffer_capacity"]
        self.lr = self.agent_prop["lr"]
        self.batch_size = self.agent_prop["batch_size"]

        self.policy_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=config_dict["DQN_prop"]["network_layers"],
        )
        self.target_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=config_dict["DQN_prop"]["network_layers"],
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())  # same weights

        self.buffer = ReplayBuffer(self.buffer_cap)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if self.device.type == 'cuda':
        #    self.policy_net.to(self.device)
        #    self.target_net.to(self.device)

        # TODO weight decay?
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), self.lr)

    def select_action(self, state):
        # Select action with epsilon-greedy strategy
        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                qs = self.policy_net(state)
                return torch.argmax(qs).item()
        return action

    def store_transition(
        self, obs_dict, action, rewards_dict, next_obs_dict, action_prob, done
    ):
        for agent_id in obs_dict.keys():
            single_obs = torch.tensor(obs_dict[agent_id], dtype=torch.float).unsqueeze(
                0
            )
            single_action = torch.tensor([[action[agent_id]]], dtype=torch.long)
            single_rewards = torch.tensor([[rewards_dict[agent_id]]], dtype=torch.float)
            single_next_obs = torch.tensor(
                next_obs_dict[agent_id], dtype=torch.float
            ).unsqueeze(0)
            self.buffer.push(single_obs, single_action, single_rewards, single_next_obs)

    def convert_batch(self, batch):
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

    def update_target_network(self):
        new_params = self.policy_net.state_dict()
        params = self.target_net.state_dict()
        for k in params.keys():
            # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
            params[k] = (1 - self.tau) * params[k] + self.tau * new_params[k]
        self.target_net.load_state_dict(params)

    def update(self):
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
        new_epsilon = self.agent_prop["epsilon_decay"]
        self.epsilon = np.maximum(new_epsilon, self.agent_prop["min_epsilon"])

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
    def __init__(self, config_dict, num_state=20, num_action=2):
        super().__init__(config_dict, num_state, num_action)

    def update(self):
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


# %% Testing

if __name__ == "__main__":
    pass

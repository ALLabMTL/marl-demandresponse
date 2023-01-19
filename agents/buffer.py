#%% Imports

import random
from collections import deque, namedtuple

import numpy as np
import torch

#%% Classes

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayBuffer:
    """Experience Replay Buffer. Pops elements out when adding more than maximum capacity.

    Args:
        capacity (int): maximum capacity of buffer, will hold (s, a, r, s') transitions.
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Add experience to memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class DDPGBuffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def push(self, obs, action, reward, next_obs, done):
        """add an experience to the memory"""
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = (
            torch.from_numpy(obs).float().to(self.device)
        )  # torch.Size([batch_size, state_dim])
        action = (
            torch.from_numpy(action).float().to(self.device)
        )  # torch.Size([batch_size, action_dim])
        reward = (
            torch.from_numpy(reward).float().to(self.device)
        )  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = (
            torch.from_numpy(next_obs).float().to(self.device)
        )  # Size([batch_size, state_dim])
        done = (
            torch.from_numpy(done).float().to(self.device)
        )  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size


#%% Testing

if __name__ == "__main__":
    state = np.ones((10, 1))
    action = np.ones(10)
    next_state = np.ones((10, 1))
    reward = np.ones(10)
    buffer = ReplayBuffer(100)
    print(buffer.memory)
    buffer.push(state, action, reward, next_state)
    buffer.push(state, action, reward, next_state)
    buffer.push(state, action, reward, next_state)
    buffer.push(state, action, reward, next_state)
    buffer.push(state, action, reward, next_state)
    buffer.push(state, action, reward, next_state)
    print(buffer.memory)

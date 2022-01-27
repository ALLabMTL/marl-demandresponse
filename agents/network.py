import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_state, num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x, temp = 1):
        x = F.relu(self.fc1(x))
        x = self.action_head(x)
        action_prob = F.softmax(x/temp, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value
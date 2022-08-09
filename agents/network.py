#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

#%% Classes


class Actor(nn.Module):
    def __init__(self, num_state, num_action, layers):
        super(Actor, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers

        self.fc = nn.ModuleList([nn.Linear(num_state, layers[0])])
        self.fc.extend(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(0, len(layers) - 1)]
        )
        self.fc.append(nn.Linear(layers[-1], num_action))
        print(self)

    def forward(self, x):
        for i in range(0, len(self.layers)):
            x = F.relu(self.fc[i](x))
        action_prob = F.softmax(self.fc[len(self.layers)](x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state, layers):
        super(Critic, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers

        self.fc = nn.ModuleList([nn.Linear(num_state, layers[0])])
        self.fc.extend(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(0, len(layers) - 1)]
        )
        self.fc.append(nn.Linear(layers[-1], 1))
        print(self)

    def forward(self, x):
        for i in range(0, len(self.layers)):
            x = F.relu(self.fc[i](x))
        value = self.fc[len(self.layers)](x)
        return value


class OldActor(nn.Module):
    def __init__(self, num_state, num_action):
        super(OldActor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)
        print(self)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.action_head(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob


class OldCritic(nn.Module):
    def __init__(self, num_state):
        super(OldCritic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

class DQN_network(nn.Module):
    def __init__(self, num_state, num_action, layers):
        super(DQN_network, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers

        self.fc = nn.ModuleList([nn.Linear(num_state, layers[0])])
        self.fc.extend(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(0, len(layers) - 1)]
        )
        self.fc.append(nn.Linear(layers[-1], num_action))
        print(self)        

    def forward(self, x):
        for i in range(0, len(self.layers)):
            x = F.relu(self.fc[i](x))
        value = self.fc[len(self.layers)](x)
        return value


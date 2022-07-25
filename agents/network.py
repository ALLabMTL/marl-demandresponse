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


"""
        # Min. 1 layer == 3-item list, e.g. [10,100,2]
        depth = len(layers) - 1
        assert depth > 1, "NN must have at least one hidden layer"

        # Architecture
        self.net = nn.Sequential()
        for n in range(depth - 1):
            self.net.add_module(f"layer_{n}", nn.Linear(layers[n], layers[n + 1]))
            self.net.add_module(f"act_{n}", nn.ReLU())
        self.net.add_module(f"layer_{n + 1}", nn.Linear(layers[n + 1], layers[n + 2]))

    def forward(self, x):
        return self.net(x)

"""


class DDPG_Actor(nn.Module):
    def __init__(self, num_state, num_actions, hidden_dim, init_w=3e-3):
        super(DDPG_Actor, self).__init__()
        self.linear1 = nn.Linear(num_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        # init weights using small numbers
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class DDPG_Critic(nn.Module):
    def __init__(self, num_state, num_actions, hidden_dim, init_w=3e-3):
        super(DDPG_Critic, self).__init__()

        self.linear1 = nn.Linear(num_state + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # init weights using small numbers
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # cat state and action along with axis 1
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


#%% Testing

if __name__ == "__main__":
    layers = [20, 100, 2]
    neuralnet = NN(layers)
    print(neuralnet)
    layers_wrong = [20, 2]
    neuralnet_wrong = NN(layers_wrong)

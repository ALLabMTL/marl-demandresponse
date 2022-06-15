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
        self.fc = [nn.Linear(num_state, layers[0])]
        for i in range(0, len(layers)-1):
            print("Actor: {}".format(i))
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
        print("Actor fc:", self.fc)    
        self.action_head = nn.Linear(layers[-1], num_action)
        print(self)

    def forward(self, x, temp=1):
        x = F.relu(self.fc[0](x))
        for i in range(1, len(self.layers)):
            x = F.relu(self.fc[i](x))
        x = self.action_head(x)
        action_prob = F.softmax(x / temp, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state, layers):
        super(Critic, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers
        self.fc = [nn.Linear(num_state, layers[0])]
        for i in range(0, len(layers)-1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
        self.state_value = nn.Linear(layers[-1], 1)

    def forward(self, x):
        x = F.relu(self.fc[0](x))
        for i in range(1, len(self.layers)):
            x = F.relu(self.fc[i](x))
        value = self.state_value(x)
        return value


class OldActor(nn.Module):
    def __init__(self, num_state, num_action):
        super(OldActor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)
        print(self)

    def forward(self, x, temp = 1):
        x = F.relu(self.fc1(x))
        x = self.action_head(x)
        action_prob = F.softmax(x/temp, dim=1)
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

class NN(nn.Module):
    def __init__(self, layers):
        super(NN, self).__init__()

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


#%% Testing

if __name__ == "__main__":
    layers = [20, 100, 2]
    neuralnet = NN(layers)
    print(neuralnet)
    layers_wrong = [20, 2]
    neuralnet_wrong = NN(layers_wrong)

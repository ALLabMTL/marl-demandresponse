#%% Imports

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from agents.network import DQN_network
from agents.buffer import ReplayBuffer, Transition

#%% Classes

class DQN:
    def __init__(self, config_dict, opt, num_state=22, num_action=2, wandb_run = None):
        super().__init__()
        self.seed = opt.net_seed
        torch.manual_seed(self.seed)
        if opt.save_actor_name:
            self.path = os.path.join(".", "actors", opt.save_actor_name)
        
        self.agent_prop = config_dict['DQN_prop']
        self.inner_layers = self.agent_prop['network_layers']
        self.gamma = self.agent_prop['gamma']
        self.tau = self.agent_prop['tau']
        self.buffer_cap = self.agent_prop['buffer_capacity']
        self.lr = self.agent_prop['lr']
        self.batch_size = self.agent_prop['batch_size']


        
        self.policy_net = DQN_network(num_state=num_state, num_action=num_action, layers=config_dict["DQN_prop"]["network_layers"])
        self.target_net = DQN_network(num_state=num_state, num_action=num_action, layers=config_dict["DQN_prop"]["network_layers"])
        self.target_net.load_state_dict(self.policy_net.state_dict()) # same weights
        
        self.buffer = ReplayBuffer(self.buffer_cap)
        
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #if self.device.type == 'cuda':
        #    self.policy_net.to(self.device)
        #    self.target_net.to(self.device)
        
        # TODO weight decay?
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), self.lr)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.policy_net.state_dict(), os.path.join(self.path, 'actor.pth'))
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.policy_net(state)
            return torch.argmax(qs).item()
            
    def store_transition(self, s, a, r, s_next):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        a = torch.tensor([[a]], dtype=torch.long)   
        r = torch.tensor([[r]], dtype=torch.float)
        s_next = torch.tensor(s_next, dtype=torch.float).unsqueeze(0)
        self.buffer.push(s, a, r, s_next)
        
    def convert_batch(self, batch):
        s = torch.cat(batch.state)
        a = torch.cat(batch.action)
        r = torch.cat(batch.reward)
        s_next = torch.cat(batch.next_state)
        #if self.device.type == 'cuda':
        #    s.to(self.device)
        #    a.to(self.device)
        #    r.to(self.device)
        #    s_next.to(self.device)
        return s, a, r, s_next
    
    def update_target_network(self):
        new_params = self.policy_net.state_dict()
        params = self.target_net.state_dict()
        for k in params.keys():
            params[k] = (1-self.tau) * params[k] + self.tau * new_params[k]
        self.target_net.load_state_dict(params)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return # exit if there are not enough transitions in buffer
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, next_state = self.convert_batch(batch)
        
        # Compute Q(s,a) from state
        q_values = self.policy_net(state).gather(1, action)
        
        # Compute max Q(s',a) from next state
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        
        # Compute expected Q(s,a)
        expected_q_values =  reward + (next_q_values.unsqueeze(1) * self.gamma)
        
        # Compute loss
        criterion = nn.SmoothL1Loss() # Huber with beta/delta=1 (default)
        loss = criterion(q_values, expected_q_values)

        # Optimize the model
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # clamp gradients
        self.policy_optimizer.step()

        self.update_target_network()


class DDQN(DQN):
    def __init__(self, config_dict, opt, num_state=20, num_action=2):
        super().__init__(config_dict, opt, num_state, num_action)
        
    def update(self):
        if len(self.buffer) < self.batch_size:
            return # exit if there are not enough transitions in buffer
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, next_state = self.convert_batch(batch)
        
        # Compute Q(s,a) from state
        q_values = self.policy_net(state).gather(1, action)
        next_action = self.policy_net(next_state).argmax(dim=1, keepdim=True)
        
        # Compute max Q(s',a) from next state
        next_q_values = self.target_net(next_state).gather(1, next_action).detach()
        
        # Compute expected Q(s,a)
        expected_q_values =  reward + (next_q_values.unsqueeze(1) * self.gamma)
        
        # Compute loss
        criterion = nn.SmoothL1Loss() # Huber with beta/delta=1 (default)
        loss = criterion(q_values, expected_q_values)

        # Optimize the model
        self.policy_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # clamp gradients
        self.policy_optimizer.step()
        
#%% Testing

if __name__ == "__main__":
    pass
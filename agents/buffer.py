#%% Imports

import numpy as np
import random
from collections import deque, namedtuple

#%% Classes

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

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

#%% Testing
   
if __name__ == "__main__":
    state = np.ones((10,1))
    action = np.ones(10) 
    next_state = np.ones((10,1))
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

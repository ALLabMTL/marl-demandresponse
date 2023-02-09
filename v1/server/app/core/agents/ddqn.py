from core.agents.dqn import DQN
from core.agents.buffer import Transition
import torch.nn as nn

class DDQN(DQN):
    def __init__(self, config_dict, opt, num_state=20, num_action=2):
        super().__init__(config_dict, opt, num_state, num_action)

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

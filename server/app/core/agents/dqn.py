import os

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field

from app.core.agents.buffer import ReplayBuffer, Transition
from app.core.agents.network import DQN_network


class DQNProperties(BaseModel):
    """Properties for DQN agent."""

    network_layers: list[int] = Field(
        default=[100, 100],
        description="List of layer sizes for the DQN network.",
    )
    gamma: float = Field(
        default=0.99,
        description="Discount factor for the reward.",
    )
    tau: float = Field(
        default=0.001,
        description="Soft target update parameter.",
    )
    lr: float = Field(
        default=3e-3,
        description="Learning rate for the DQN network.",
    )
    buffer_capacity: int = Field(
        default=524288,
        description="Capacity of the replay buffer.",
    )
    batch_size: int = Field(
        default=256,
        description="Batch size for the DQN agent.",
    )
    epsilon_decay: float = Field(
        default=0.99998,
        description="Epsilon decay rate for the DQN agent.",
    )
    min_epsilon: float = Field(
        default=0.01,
        description="Minimum epsilon for the DQN agent.",
    )


from app.core.agents.agent import Agent


class DQN:
    def __init__(
        self,
        static_props: DQNProperties,
        opt,
        num_state=22,
        num_action=2,
        wandb_run=None,
    ):
        super().__init__()
        self.seed = opt.net_seed
        torch.manual_seed(self.seed)
        if opt.save_actor_name:
            self.path = os.path.join(".", "actors", opt.save_actor_name)

        self.static_props = static_props

        self.policy_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=static_props.network_layers,
        )
        self.target_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=static_props.network_layers,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())  # same weights

        self.buffer = ReplayBuffer(self.static_props.buffer_capacity)

        # TODO weight decay?
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), self.static_props.lr
        )

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.policy_net.state_dict(), os.path.join(self.path, "actor.pth"))

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
        return s, a, r, s_next

    def update_target_network(self):
        new_params = self.policy_net.state_dict()
        params = self.target_net.state_dict()
        for k in params.keys():
            # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
            params[k] = (1 - self.static_props.tau) * params[
                k
            ] + self.static_props.tau * new_params[k]
        self.target_net.load_state_dict(params)

    def update(self):
        if len(self.buffer) < self.static_props.batch_size:
            return  # exit if there are not enough transitions in buffer

        transitions = self.buffer.sample(self.static_props.batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, next_state = self.convert_batch(batch)

        # Compute Q(s,a) from state
        q_values = self.policy_net(state).gather(1, action)

        # Compute max Q(s',a) from next state
        next_q_values = self.target_net(next_state).max(1)[0].detach()

        # Compute expected Q(s,a)
        expected_q_values = reward + (
            next_q_values.unsqueeze(1) * self.static_props.gamma
        )

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


class DQNAgent(Agent):
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        self.id = agent_properties["id"]
        self.agent_name = agent_properties["actor_name"]
        self.agent_path = os.path.join(".", "actors", self.agent_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.DQN_net = DQN_network(
            num_state=num_state,
            num_action=num_action,
            layers=config_dict["DQN_prop"]["network_layers"],
        )
        self.DQN_net.load_state_dict(
            torch.load(os.path.join(self.agent_path, "DQN.pth"))
        )
        self.DQN_net.eval()

    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.DQN_net(state)
        action = torch.argmax(qs).item()
        return action

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, Field
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from app.core.agents.network import Actor, Critic
from app.utils.logger import logger


class PPOProperties(BaseModel):
    """Properties for PPO agent."""

    actor_layers: list[int] = Field(
        default=[100, 100],
        description="List of layer sizes for the actor network.",
    )
    critic_layers: list[int] = Field(
        default=[100, 100],
        description="List of layer sizes for the critic network.",
    )
    gamma: float = Field(
        default=0.99,
        description="Discount factor for the reward.",
    )
    lr_critic: float = Field(
        default=3e-3,
        description="Learning rate for the critic network.",
    )
    lr_actor: float = Field(
        default=3e-3,
        description="Learning rate for the actor network.",
    )
    clip_param: float = Field(
        default=0.2,
        description="Clipping parameter for the PPO loss.",
    )
    max_grad_norm: float = Field(
        default=0.5,
        description="Maximum norm for the gradient clipping.",
    )
    ppo_update_time: int = Field(
        default=10,
        description="Update time for the PPO agent.",
    )
    batch_size: int = Field(
        default=256,
        description="Batch size for the PPO agent.",
    )
    zero_eoepisode_return: bool = Field(
        default=False,
        # description="Whether to zero the episode return when the episode ends.",
    )


class PPO:
    def __init__(
        self,
        static_props: PPOProperties,
        num_state: int = 22,
        num_action: int = 2,
        seed: int = 1,
        wandb_run=None,
    ):
        super(PPO, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.static_props = static_props
        # if True:
        #    self.actor_net = OldActor(num_state=num_state, num_action=num_action)
        #    self.critic_net = OldCritic(num_state=num_state)
        self.actor_net = Actor(
            num_state=num_state,
            num_action=num_action,
            layers=static_props.actor_layers,
        ).to(self.device)
        self.critic_net = Critic(
            num_state=num_state, layers=static_props.critic_layers
        ).to(self.device)
        # TODO: change this static value (only for tests)
        self.nb_agents = 10

        # Initialize buffer
        self.buffer = {}
        for agent in range(self.nb_agents):
            self.buffer[agent] = []

        logger.info(
            "ppo_update_time: {}, max_grad_norm: {}, clip_param: {}, gamma: {}, batch_size: {}, lr_actor: {}, lr_critic: {}".format(
                self.static_props.ppo_update_time,
                self.static_props.max_grad_norm,
                self.static_props.clip_param,
                self.static_props.gamma,
                self.static_props.batch_size,
                self.static_props.lr_actor,
                self.static_props.lr_critic,
            )
        )
        self.training_step = 0

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), self.static_props.lr_actor
        )
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), self.static_props.lr_critic
        )

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            # pylint: disable=not-callable
            action_prob = self.actor_net(state)
        c = Categorical(action_prob.cpu())
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        # state = torch.from_numpy(state)
        state = state.to(self.device)
        with torch.no_grad():
            # pylint: disable=not-callable
            value = self.critic_net(state)
        return value.cpu().item()

    def reset_buffer(self):
        self.buffer = {}
        for agent in range(self.nb_agents):
            self.buffer[agent] = []

    def store_transition(self, transition, agent):
        self.buffer[agent].append(transition)

    def update(self, t):
        sequential_buffer = []
        for agent in range(self.nb_agents):
            sequential_buffer += self.buffer[agent]

        state_np = np.array([t.state for t in sequential_buffer])
        next_state_np = np.array([t.next_state for t in sequential_buffer])
        action_np = np.array([t.action for t in sequential_buffer])
        old_action_log_prob_np = np.array([t.a_log_prob for t in sequential_buffer])

        state = torch.tensor(state_np, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state_np, dtype=torch.float).to(self.device)
        action = torch.tensor(action_np, dtype=torch.long).view(-1, 1).to(self.device)
        reward = [t.reward for t in sequential_buffer]
        old_action_log_prob = (
            torch.tensor(old_action_log_prob_np, dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        done = [t.done for t in sequential_buffer]

        """
        # Changed to accelerate process. UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray $

        state = torch.tensor([t.state for t in sequential_buffer], dtype=torch.float) 
        next_state = torch.tensor([t.next_state for t in sequential_buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in sequential_buffer], dtype=torch.long).view(-1, 1) 
        reward = [t.reward for t in sequential_buffer] 

        old_action_log_prob = torch.tensor( 
            [t.a_log_prob for t in sequential_buffer], dtype=torch.float 
        ).view(-1, 1) 
        done = [t.done for t in sequential_buffer] 
        
        """

        Gt = []
        for i in reversed(range(len(reward))):
            if done[i]:
                if self.static_props.zero_eoepisode_return:
                    R = 0
                else:
                    R = self.get_value(
                        next_state[i]
                    )  # When last state of episode, start from estimated value of next state
            R = reward[i] + self.static_props.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(self.device)
        ratios = np.array([])
        clipped_ratios = np.array([])
        gradient_norms = np.array([])
        logger.info("The agent is updating....")
        for i in range(self.static_props.ppo_update_time):
            for index in BatchSampler(
                SubsetRandomSampler(range(len(sequential_buffer))),
                self.static_props.batch_size,
                False,
            ):
                if self.training_step % 1000 == 0:
                    logger.info(
                        "Time step: {} ，train {} times".format(t, self.training_step)
                    )
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)

                # pylint: disable=not-callable
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core
                # pylint: disable=not-callable
                action_prob = self.actor_net(state[index]).gather(
                    1, action[index]
                )  # new policy
                ratio = action_prob / old_action_log_prob[index]
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.static_props.clip_param,
                    1 + self.static_props.clip_param,
                )
                ratios = np.append(ratios, ratio.cpu().detach().numpy())
                clipped_ratios = np.append(
                    clipped_ratios, clipped_ratio.cpu().detach().numpy()
                )

                surr1 = ratio * advantage
                surr2 = clipped_ratio * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.static_props.max_grad_norm
                )
                gradient_norms = np.append(gradient_norms, gradient_norm.cpu().detach())
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.static_props.max_grad_norm
                )
                self.critic_net_optimizer.step()
                self.training_step += 1

        self.reset_buffer()  # clear experience

from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, Field
from torch import Tensor

from app.core.agents.network import DDPG_Network


def copy_model(src, dst):
    for src_param, dst_param in zip(src.parameters(), dst.parameters()):
        dst_param.data.copy_(src_param.data)
    return dst


# TODO(Victor): ddpg, mappo, ppo,


class DDPGProperties(BaseModel):
    """Properties for MAPPO agent."""

    actor_hidden_dim: int = Field(
        default=256,
        description="Hidden dimension for the actor network.",
    )
    critic_hidden_dim: int = Field(
        default=256,
        description="Hidden dimension for the critic network.",
    )
    lr_critic: float = Field(
        default=3e-3,
        description="Learning rate for the critic network.",
    )
    lr_actor: float = Field(
        default=3e-3,
        description="Learning rate for the actor network.",
    )
    soft_tau: float = Field(
        default=0.01,
        description="Soft target update parameter.",
    )
    clip_param: float = Field(
        default=0.2,
        description="Clipping parameter for the PPO loss.",
    )
    max_grad_norm: float = Field(
        default=0.5,
        description="Maximum norm for the gradient clipping.",
    )
    ddpg_update_time: int = Field(
        default=10,
        description="Update time for the DDPG agent.",
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for the DDPG agent.",
    )
    buffer_capacity: int = Field(
        default=524288,
        description="Capacity of the replay buffer.",
    )
    episode_num: int = Field(
        default=10000,
        # description="Number of episodes for the MAPPO agent.",
    )
    learn_interval: int = Field(
        default=100,
        description="Learning interval for the MAPPO agent.",
    )
    random_steps: int = Field(
        default=100,
        # description="Number of random steps for the MAPPO agent.",
    )
    gumbel_softmax_tau: float = Field(
        default=1.0,
        description="Temperature for the gumbel softmax distribution.",
    )
    DDPG_shared: bool = Field(
        default=True,
        # description="Whether to use the shared DDPG network.",
    )
    gamma: float = Field(
        default=0.99,
        description="Discount factor for the reward.",
    )


class DDPG:
    def __init__(
        self,
        static_props: DDPGProperties,
        opt,
        num_all_state_action=24,
        num_state=22,
        num_action=2,
        wandb_run=None,
        import_nets=None,
    ):
        """
        This is the main class for the DDPG agent.
        @param config_dict: dictionary of parameters for the agent
        @param opt: the option class
        @param num_state: number of state variables
        @param num_action: number of action variables
        @param wandb_run: if wandb is used, this is the wandb run
        @param import_nets: if the agent is imported from a predetermined model,
                this is a dictionary of the networks. Used for parameter sharing.
        """
        super(DDPG, self).__init__()

        self.seed = opt.net_seed
        torch.manual_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initilize the actor and critic network
        self.actor_net = DDPG_Network(
            in_dim=num_state,
            out_dim=num_action,
            hidden_dim=static_props.actor_hidden_dim,
        )
        self.critic_net = DDPG_Network(
            in_dim=num_all_state_action,
            out_dim=1,
            hidden_dim=static_props.critic_hidden_dim,
        )

        # initialize the target actor and critic network
        self.tgt_actor_net = DDPG_Network(
            in_dim=num_state,
            out_dim=num_action,
            hidden_dim=static_props.actor_hidden_dim,
        )
        self.tgt_critic_net = DDPG_Network(
            in_dim=num_all_state_action,
            out_dim=1,
            hidden_dim=static_props.critic_hidden_dim,
        )

        if import_nets is not None:
            for key in import_nets:
                try:
                    self.__dict__[key] = import_nets[key]
                except KeyError:
                    print("KeyError: {}".format(key))

        # copy the target network from the actor and critic network
        copy_model(self.actor_net, self.tgt_actor_net)
        copy_model(self.critic_net, self.tgt_critic_net)

        # initialize the optimizer
        self.lr_actor = static_props.lr_actor
        self.lr_critic = static_props.lr_critic
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), self.lr_critic)

        # other params
        self.soft_tau = static_props.soft_tau  # soft update for target network
        self.gumbel_softmax_tau = static_props.gumbel_softmax_tau
        self.gamma = static_props.gamma
        self.batch_size = static_props.batch_size
        self.wandb_run = wandb_run
        self.log_wandb = not opt.no_wandb

        # initialize the buffer
        self.buffer = []

        self.ddpg_update_time = static_props.ddpg_update_time
        self.max_grad_norm = static_props.max_grad_norm
        self.clip_param = static_props.clip_param

        print(
            "ddpg_update_time: {}, max_grad_norm: {}, clip_param: {}, gamma: {}, batch_size: {}, lr_actor: {}, lr_critic: {}".format(
                self.ddpg_update_time,
                self.max_grad_norm,
                self.clip_param,
                self.gamma,
                self.batch_size,
                self.lr_actor,
                self.lr_critic,
            )
        )
        self.counter = 0
        self.training_step = 0

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def select_action(self, state, output_logits=False, is_target=False):
        if not is_target:
            logits = self.actor_net(state)  # torch.Size([batch_size, action_size])
        else:
            logits = self.tgt_actor_net(state)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, tau=self.gumbel_softmax_tau, hard=True)
        action = action.squeeze(0).detach() if is_target else action
        if output_logits:
            return action, logits
        return action

    def get_value(
        self, state_list: List[Tensor], act_list: List[Tensor], is_target=False
    ):
        x = torch.cat(state_list + act_list, 1)
        return (
            self.tgt_critic_net(x).squeeze(1)
            if is_target
            else self.critic_net(x).squeeze(1)
        )  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.critic_optimizer.step()

    @staticmethod
    def soft_update(from_network, to_network, soft_tau):
        """copy the parameters of `from_network` to `to_network` with a proportion of tau"""
        for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
            to_p.data.copy_(soft_tau * from_p.data + (1.0 - soft_tau) * to_p.data)

    def update_target(self):
        self.soft_update(self.actor_net, self.tgt_actor_net, self.soft_tau)
        self.soft_update(self.critic_net, self.tgt_critic_net, self.soft_tau)


def get_dim_info(opt, n_state, n_action=2):
    """get the dimension information of the environment"""
    dim_info = {}
    for agent_id in range(opt.nb_agents):
        dim_info[agent_id] = []
        dim_info[agent_id].append(n_state)
        dim_info[agent_id].append(n_action)
    return dim_info

from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, Field
from torch import Tensor

from app.core.agents.buffer import DDPGBuffer as Buffer
from app.core.agents.trainables.network import DDPG_Network


def copy_model(src, dst):
    for src_param, dst_param in zip(src.parameters(), dst.parameters()):
        dst_param.data.copy_(src_param.data)
    return dst


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
    ) -> None:
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

    def update_actor(self, loss) -> None:
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss) -> None:
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.critic_optimizer.step()

    @staticmethod
    def soft_update(from_network, to_network, soft_tau) -> None:
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


class MADDPGPropreties(DDPGProperties):
    pass


class MADDPG:
    def __init__(
        self, static_props: MADDPGPropreties, opt, num_state, wandb_run
    ) -> None:
        # sum all the dims of each agent to get input dim for critic
        # global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        self.opt = opt
        self.num_state = num_state
        self.static_props = static_props
        self.capacity = static_props.buffer_capacity
        dim_info = get_dim_info(opt, num_state)
        global_state_action_dim = sum(sum(val) for val in dim_info.values())
        for agent_id, (state_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = DDPG(
                static_props,
                self.opt,
                global_state_action_dim,
                state_dim,
                action_dim,
            )
            self.buffers[agent_id] = Buffer(
                self.capacity,
                state_dim,
                action_dim,
                "cpu",
            )
        self.shared = opt.DDPG_shared
        assert self.shared != -1, "shared must be set as True or False (1 or 0)"
        print("DDPG shared status: {}".format(self.shared))
        if self.shared:
            self.agents[agent_id].actor_net = self.agents[0].actor_net
            self.agents[agent_id].critic_net = self.agents[0].critic_net
            self.agents[agent_id].tgt_actor_net = self.agents[0].tgt_actor_net
            self.agents[agent_id].tgt_critic_net = self.agents[0].tgt_critic_net
            self.agents[agent_id].actor_optimizer = self.agents[0].actor_optimizer
            self.agents[agent_id].critic_optimizer = self.agents[0].critic_optimizer

        self.dim_info = dim_info
        self.wandb_run = wandb_run
        self.batch_size = self.static_props.batch_size
        self.result_dir = os.path.join(
            "./ddpg_results"
        )  # directory to save the training result
        self.logger = self.setup_logger(os.path.join(self.result_dir, "maddpg.log"))
        self.log_wandb = not opt.no_wandb

    def setup_logger(self, filename):
        """set up logger with filename."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(filename, mode="w")
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s--%(levelname)s--%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        return logger

    def push(self, state, action, reward, next_state, done):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in state.keys():
            s = state[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]

            r = reward[agent_id]
            next_s = next_state[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].push(s, a, r, next_s, d)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        # obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
        state, action, reward, next_state, done, next_action = {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            s, a, r, n_s, d = buffer.sample(indices)
            state[agent_id] = s
            action[agent_id] = a
            reward[agent_id] = r
            next_state[agent_id] = n_s
            done[agent_id] = d
            # calculate next_action using target_network and next_state
            next_action[agent_id] = self.agents[agent_id].select_action(
                n_s, is_target=True
            )

        # return obs, act, reward, next_obs, done, next_act
        return state, action, reward, next_state, done, next_action

    def select_action(self, state):
        # select action for each agent
        actions = {}
        for agent, s in state.items():
            s = torch.from_numpy(s).unsqueeze(0).float()
            a = self.agents[agent].select_action(s)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            self.logger.info(f"{agent} action: {actions[agent]}")
        return actions

    def update_target(self) -> None:
        # update target network for all the agents
        for agent in self.agents.values():
            agent.update_target()

    def update(self) -> None:
        for agent_id, agent in self.agents.items():
            state, action, reward, next_state, done, next_action = self.sample(
                self.batch_size
            )

            # update critic
            critic_value = agent.get_value(
                list(state.values()), list(action.values()), is_target=False
            )

            # calculate target critic value
            next_target_critic_value = agent.get_value(
                list(next_state.values()), list(next_action.values()), is_target=True
            )
            target_value = reward[agent_id] + self.agents[
                agent_id
            ].gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(
                critic_value, target_value.detach(), reduction="mean"
            )
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action_, logits = agent.select_action(state[agent_id], output_logits=True)
            # print("action", action)
            # print("agent_id", agent_id)
            action[agent_id] = action_
            actor_loss = -agent.get_value(
                list(state.values()), list(action.values())
            ).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
        if self.log_wandb:
            self.wandb_run.log({"critic_loss": critic_loss, "actor_loss": actor_loss})

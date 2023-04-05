from collections import namedtuple
from copy import deepcopy
import os
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from app.core.agents.trainables.ppo import PPOProperties
from app.core.agents.trainables.network import Actor, Critic
from app.core.agents.trainables.trainable import Trainable


class MAPPOProperties(PPOProperties):
    pass


Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "others_actions",
        "a_log_prob",
        "reward",
        "next_state",
        "done",
    ],
)


class MAPPO(Trainable):
    def __init__(
        self, config: PPOProperties, num_state=22, num_action=2, seed=1
    ) -> None:
        super(MAPPO, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        self.last_actions: Dict[int, bool] = {}
        self.last_probs: Dict[int, float] = {}
        self.actor_net = Actor(
            num_state=num_state,
            num_action=num_action,
            layers=config.actor_layers,
        )
        self.critic_net = Critic(
            num_state=num_state + num_action - 1,
            layers=config.critic_layers,
        )

        self.buffer: List[Transition] = []
        self.batch_size = config.batch_size
        self.ppo_update_time = config.ppo_update_time
        self.max_grad_norm = config.max_grad_norm
        self.clip_param = config.clip_param
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic

        print(
            "mappo_update_time: {}, max_grad_norm: {}, clip_param: {}, gamma: {}, batch_size: {}, lr_actor: {}, lr_critic: {}".format(
                self.ppo_update_time,
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

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), self.lr_critic
        )

    def select_actions(self, observations: List[np.ndarray]) -> Dict[int, bool]:
        actions: Dict[int, bool] = {}
        probs: Dict[int, float] = {}
        for obs_id, obs in enumerate(observations):
            tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                # pylint: disable=not-callable
                action_prob = self.actor_net(tensor)
            c = Categorical(action_prob.cpu())
            action = c.sample()
            actions[obs_id] = action.item()
            probs[obs_id] = action_prob[:, action.item()].item()
        self.last_probs = probs
        self.last_actions = actions
        return actions

    # def get_value(self, state):
    #    state = torch.from_numpy(state)
    #    with torch.no_grad():
    #        value = self.critic_net(state)
    #    return value.item()

    def store_transition(
        self,
        observations: List[np.ndarray],
        next_observations: List[np.ndarray],
        rewards: Dict[int, float],
        done: bool,
    ) -> None:
        for observation_id, next_observation in enumerate(next_observations):
            action_k = deepcopy(self.last_actions[observation_id])
            action_k.pop(observation_id)
            other_action_list = list(action_k.values())

            transition = Transition(
                observations[observation_id],
                self.last_actions[observation_id],
                other_action_list,
                self.last_probs[observation_id],
                rewards[observation_id],
                next_observation,
                done,
            )
            self.buffer[observation_id].append(transition)
            self.buffer.append(transition)
            self.counter += 1

    def update(self, t) -> None:
        if len(self.buffer) < self.batch_size:
            return
        else:
            print("UPDATING")

            state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
            action = torch.tensor(
                [t.action for t in self.buffer], dtype=torch.long
            ).view(-1, 1)
            others_actions = torch.tensor(
                [t.others_actions for t in self.buffer], dtype=torch.long
            )
            reward = [t.reward for t in self.buffer]
            old_action_log_prob = torch.tensor(
                [t.a_log_prob for t in self.buffer], dtype=torch.float
            ).view(-1, 1)
            done = [t.done for t in self.buffer]
            R = 0
            Gt: list = []
            for i in reversed(range(len(reward))):
                if done[i]:
                    R = 0
                R = reward[i] + self.gamma * R
                Gt.insert(0, R)
            Gt_tensor = torch.tensor(Gt, dtype=torch.float)
            ratios = np.array([])
            clipped_ratios = np.array([])
            gradient_norms = np.array([])
            # print("The agent is updateing....")
            for i in range(self.ppo_update_time):
                for index in BatchSampler(
                    SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False
                ):
                    if self.training_step % 1000 == 0:
                        print(
                            "Time step: {} ，train {} times".format(
                                t, self.training_step
                            )
                        )
                    # with torch.no_grad():
                    Gt_index = Gt_tensor[index].view(-1, 1)

                    V = self.critic_net(
                        torch.cat((state[index], others_actions[index]), dim=1)
                    )
                    delta = Gt_index - V
                    advantage = delta.detach()

                    # epoch iteration, PPO core
                    action_prob = self.actor_net(state[index]).gather(
                        1, action[index]
                    )  # new policy
                    ratio = action_prob / old_action_log_prob[index]
                    clipped_ratio = torch.clamp(
                        ratio, 1 - self.clip_param, 1 + self.clip_param
                    )

                    ratios = np.append(ratios, ratio.detach().numpy())
                    clipped_ratios = np.append(
                        clipped_ratios, clipped_ratio.detach().numpy()
                    )

                    surr1 = ratio * advantage
                    surr2 = clipped_ratio * advantage

                    # update actor network
                    action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                    # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                    self.actor_optimizer.zero_grad()
                    action_loss.backward()
                    gradient_norm = nn.utils.clip_grad_norm_(
                        self.actor_net.parameters(), self.max_grad_norm
                    )
                    gradient_norms = np.append(gradient_norms, gradient_norm.detach())
                    self.actor_optimizer.step()

                    # update critic network
                    value_loss = F.mse_loss(Gt_index, V)
                    # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                    self.critic_net_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic_net.parameters(), self.max_grad_norm
                    )
                    self.critic_net_optimizer.step()
                    self.training_step += 1

            del self.buffer[:]  # clear experience

    def save(self, path: str, time_step=None) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        actor_net = self.actor_net
        if time_step:
            torch.save(
                actor_net.state_dict(),
                os.path.join(path, "actor" + str(time_step) + ".pth"),
            )
        else:
            torch.save(actor_net.state_dict(), os.path.join(path, "actor.pth"))

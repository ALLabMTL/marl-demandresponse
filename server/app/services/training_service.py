import random
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from time import sleep
from typing import Dict, List, Union
import json
import numpy as np
import torch

from app.core.agents.agent import Agent
from app.core.agents.ppo import PPO
from app.core.config.config import MarlConfig
from app.core.environment.environment import Environment
from app.utils.logger import logger
from app.utils.metrics import Metrics
from app.utils.utils import normStateDict

from .client_manager_service import ClientManagerService
from .socket_manager_service import SocketManager


class TrainingService:
    env: Environment
    obs_dict: Dict[int, List[Union[float, str, bool, datetime]]]
    nb_time_steps: int
    nb_test_logs: int
    nb_tr_logs: int
    nb_tr_epochs: int
    agent: Agent
    num_state: int
    metrics: Metrics
    time_steps_per_episode: int
    time_steps_per_epoch: int
    time_steps_train_log: int
    time_steps_test_log: int
    transition = namedtuple(
        "Transition", ["state", "action", "a_log_prob", "reward", "next_state", "done"]
    )

    stop: bool

    def load_config(self) -> MarlConfig:
        # with open("MarlConfig.json", "r") as f:
        #     config_dict = MarlConfig(**json.load(f))
        # return config_dict
        logger.warning("TODO(Victor): Loading default config")
        return MarlConfig()

    def __init__(
        self,
        socket_manager_service: SocketManager,
        client_manager_service: ClientManagerService,
    ) -> None:
        self.client_manager_service = client_manager_service
        self.socket_manager_service = socket_manager_service
        self.stop = False

        self.config: MarlConfig = self.load_config()

    def initialize(self) -> None:
        random.seed(1)
        self.env = Environment(self.config.env_prop)
        self.metrics = Metrics()
        # TODO: Get these arguments from config file (parser)
        self.nb_time_steps = 1000
        self.nb_test_logs = 100
        self.nb_tr_logs = 100
        self.nb_tr_epochs = 20
        self.obs_dict = self.env._reset()
        self.num_state = len(
            normStateDict(self.obs_dict[next(iter(self.obs_dict))], self.config)
        )
        # TODO: Get agent from config file
        self.agent = PPO(self.config.PPO_prop, self.num_state)
        self.time_steps_per_episode = int(self.nb_time_steps / self.nb_tr_epochs)
        self.time_steps_per_epoch = int(self.nb_time_steps / self.nb_tr_epochs)
        self.time_steps_train_log = int(self.nb_time_steps / self.nb_tr_logs)
        self.time_steps_test_log = int(self.nb_time_steps / self.nb_test_logs)
        logger.info(f"Number of states: {self.num_state}")

    async def train(self):
        # Initialize training variables
        logger.info(f"Initializing environment...")
        await self.socket_manager_service.emit(
            "success", {"message": "Initializing environment..."}
        )

        self.initialize()
        self.client_manager_service.initialize_data()

        await self.socket_manager_service.emit(
            "success", {"message": "Starting simulation"}
        )
        self.obs_dict = self.env._reset()

        for t in range(self.nb_time_steps):
            if self.stop:
                logger.info(f"Training stopped at time {t}")
                await self.socket_manager_service.emit("stopped", {})
                break

            (
                data_messages,
                houses_messages,
            ) = self.client_manager_service.update_data_change(self.obs_dict)

            await self.socket_manager_service.emit("houseChange", houses_messages)
            await self.socket_manager_service.emit("dataChange", data_messages)

            sleep(2)

            # Select action with probabilities
            action_and_prob = {
                k: self.agent.select_action(
                    normStateDict(self.obs_dict[k], self.config)
                )
                for k in self.obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in self.obs_dict.keys()}
            action_prob = {k: action_and_prob[k][1] for k in self.obs_dict.keys()}

            # Take action and get new transition
            next_obs_dict, rewards_dict = self.env._step(action)

            # Episode is done
            done = t % self.time_steps_per_episode == self.time_steps_per_episode - 1

            # Storing in replay buffer
            for k in self.obs_dict.keys():
                self.agent.store_transition(
                    self.transition(
                        normStateDict(self.obs_dict[k], self.config),
                        action[k],
                        action_prob[k],
                        rewards_dict[k],
                        normStateDict(next_obs_dict[k], self.config),
                        done,
                    ),
                    k,
                )
                # Update metrics
                self.metrics.update(
                    k, self.obs_dict, next_obs_dict, rewards_dict, self.env
                )

            # Set next state as current state
            self.obs_dict = next_obs_dict

            # New episode, reset environment
            if done:
                logger.info(f"New episode at time {t}")
                await self.socket_manager_service.emit(
                    "success", {"message": f"New episode at time {t}"}
                )

                self.obs_dict = self.env._reset()

            # Epoch: update agent
            if (
                t % self.time_steps_per_epoch == self.time_steps_per_epoch - 1
                and len(self.agent.buffer[0]) >= self.agent.batch_size
            ):
                logger.info(f"Updating agent at time {t}")
                await self.socket_manager_service.emit(
                    "success", {"message": f"Updating agent at time {t}"}
                )

                self.agent.update(t)

            # Log train statistics
            if (
                t % self.time_steps_train_log == self.time_steps_train_log - 1
            ):  # Log train statistics
                logger.info(f"Logging stats at time {t}")
                await self.socket_manager_service.emit(
                    "success", {"message": f"Logging stats at time {t}"}
                )

                logged_metrics = self.metrics.log(t, self.time_steps_train_log)
                logger.info(f"Stats : {logged_metrics}")

                self.metrics.reset()

            # Test policy
            if (
                t % self.time_steps_test_log == self.time_steps_test_log - 1
            ):  # Test policy
                logger.info(f"Testing at time {t}")
                await self.socket_manager_service.emit(
                    "success", {"message": f"Testing at time {t}"}
                )
                # metrics_test = self.test_ppo_agent(t)
                # logger.info(f"Metrics test: {metrics_test}")
                # logger.info("Training step - {}".format(t))

        logger.info("Simulation ended")
        await self.socket_manager_service.emit("stopped", {})
        await self.socket_manager_service.emit(
            "success", {"message": "Simulation ended"}
        )

    def test_agent(self, tr_time_steps) -> dict:
        """
        Test ppo agent on an episode of nb_test_timesteps, with
        """
        nb_time_steps_test = 10
        env = deepcopy(self.env)
        cumul_avg_reward = 0
        cumul_temp_error = 0
        cumul_signal_error = 0
        obs_dict = self.env._reset()
        nb_agents = len(env.cluster.buildings)
        with torch.no_grad():
            for t in range(nb_time_steps_test):
                action_and_prob = {
                    k: self.agent.select_action(normStateDict(obs_dict[k], config_dict))
                    for k in obs_dict.keys()
                }
                action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
                obs_dict, rewards_dict = env._step(action)
                for i in range(nb_agents):
                    cumul_avg_reward += rewards_dict[i] / nb_agents
                    cumul_temp_error += (
                        np.abs(obs_dict[i]["indoor_temp"] - obs_dict[i]["target_temp"])
                        / nb_agents
                    )
                    cumul_signal_error += np.abs(
                        obs_dict[i]["reg_signal"] - obs_dict[i]["cluster_hvac_power"]
                    ) / (nb_agents**2)
        mean_avg_return = cumul_avg_reward / nb_time_steps_test
        mean_temp_error = cumul_temp_error / nb_time_steps_test
        mean_signal_error = cumul_signal_error / nb_time_steps_test

        return {
            "Mean test return": mean_avg_return,
            "Test mean temperature error": mean_temp_error,
            "Test mean signal error": mean_signal_error,
            "Training steps": tr_time_steps,
        }

    async def train_ppo(self):
        random.seed(1)
        env = Environment()
        obs_dict = env._reset()
        num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
        logger.info(f"Number of states: {num_state}")
        agent = PPO(config_dict, num_state)
        # TODO: take these arguments from app.config file
        nb_time_steps = 1000
        nb_test_logs = 100
        nb_tr_logs = 100
        nb_tr_epochs = 20
        # Initialize variables
        Transition = namedtuple(
            "Transition",
            ["state", "action", "a_log_prob", "reward", "next_state", "done"],
        )
        time_steps_per_episode = int(nb_time_steps / nb_tr_epochs)
        time_steps_per_epoch = int(nb_time_steps / nb_tr_epochs)
        time_steps_train_log = int(nb_time_steps / nb_tr_logs)
        time_steps_test_log = int(nb_time_steps / nb_test_logs)
        metrics = Metrics()

        # Get first observation
        obs_dict = env._reset()

        for t in range(nb_time_steps):
            await self.client_manager_service.emit_data_change(obs_dict)
            sleep(2)
            # Select action with probabilities
            action_and_prob = {
                k: agent.select_action(normStateDict(obs_dict[k], config_dict))
                for k in obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
            action_prob = {k: action_and_prob[k][1] for k in obs_dict.keys()}

            # Take action and get new transition
            next_obs_dict, rewards_dict = env._step(action)

            # Episode is done
            done = t % time_steps_per_episode == time_steps_per_episode - 1

            # Storing in replay buffer
            for k in obs_dict.keys():
                agent.store_transition(
                    Transition(
                        normStateDict(obs_dict[k], config_dict),
                        action[k],
                        action_prob[k],
                        rewards_dict[k],
                        normStateDict(next_obs_dict[k], config_dict),
                        done,
                    ),
                    k,
                )
                # Update metrics
                metrics.update(k, obs_dict, next_obs_dict, rewards_dict, env)

            # Set next state as current state
            obs_dict = next_obs_dict

            # New episode, reset environment
            if done:
                logger.info(f"New episode at time {t}")
                obs_dict = env._reset()

            # Epoch: update agent
            if (
                t % time_steps_per_epoch == time_steps_per_epoch - 1
                and len(agent.buffer[0]) >= agent.batch_size
            ):
                logger.info(f"Updating agent at time {t}")
                agent.update(t)

            # Log train statistics
            if (
                t % time_steps_train_log == time_steps_train_log - 1
            ):  # Log train statistics
                logger.info("Logging stats at time {}".format(t))
                logged_metrics = metrics.log(t, time_steps_train_log)
                logger.info(f"Stats : {logged_metrics}")

                metrics.reset()

            # Test policy
            if t % time_steps_test_log == time_steps_test_log - 1:  # Test policy
                logger.info(f"Testing at time {t}")
                # metrics_test = self.test_ppo_agent(agent, env, t)
                # logger.info(f"Metrics test: {metrics_test}")
                # logger.info("Training step - {}".format(t))

        logger.info("Simulation ended")

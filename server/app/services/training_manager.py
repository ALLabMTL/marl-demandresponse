import os
import random
from copy import deepcopy
from datetime import datetime
from time import sleep
from typing import Dict, List, Union

import numpy as np
import torch

from app.config import config_dict
from app.core.agents.ppo import PPO
from app.core.environment.environment import Environment
from app.services.experiment import Experiment
from app.services.metrics_service import Metrics
from app.services.parser_service import MarlConfig, ParserService
from app.utils.logger import logger
from app.utils.utils import normStateDict

from .client_manager_service import ClientManagerService
from .socket_manager_service import SocketManager


class TrainingManager(Experiment):
    env: Environment
    obs_dict: dict
    stop: bool

    def __init__(
        self,
        socket_manager_service: SocketManager,
        client_manager_service: ClientManagerService,
        metrics_service: Metrics,
        parser_service: ParserService,
    ) -> None:
        self.client_manager_service = client_manager_service
        self.socket_manager_service = socket_manager_service
        self.metrics_service = metrics_service
        self.stop = False
        self.static_config = parser_service.config
        self.initialize(parser_service.config)

    def initialize(self, marl_config: MarlConfig) -> None:
        random.seed(1)
        self.env = Environment(marl_config.env_prop)
        self.nb_agents = self.env.cluster.nb_agents
        # TODO: Get these arguments from config file (parser)
        self.nb_time_steps = 1000
        self.nb_time_steps_test = 300
        self.nb_test_logs = 100
        self.save_actor_name: str = "PPO"
        self.nb_tr_logs = 100
        self.nb_tr_epochs = 20
        self.nb_tr_episodes = 2
        self.speed: int = 2
        self.time_steps_per_saving_actor = 2

        self.obs_dict = self.env._reset()
        self.num_state = len(
            normStateDict(self.obs_dict[next(iter(self.obs_dict))], config_dict)
        )
        # TODO: Get agent from config file
        self.agent = PPO(config_dict, self.num_state)
        self.time_steps_per_episode = int(self.nb_time_steps / self.nb_tr_episodes)
        self.time_steps_per_epoch = int(self.nb_time_steps / self.nb_tr_epochs)
        self.time_steps_train_log = int(self.nb_time_steps / self.nb_tr_logs)
        self.time_steps_test_log = int(self.nb_time_steps / self.nb_test_logs)
        logger.info("Number of states: %d", self.num_state)

    async def start(self) -> None:
        # Initialize training variables
        logger.info("Initializing environment...")
        await self.socket_manager_service.emit(
            "success", {"message": "Initializing environment..."}
        )

        self.initialize(self.static_config)
        self.client_manager_service.initialize_data()
        await self.socket_manager_service.emit("agent", self.agent_name)

        await self.socket_manager_service.emit(
            "success", {"message": "Starting simulation"}
        )
        self.obs_dict = self.env._reset()

        for step in range(self.nb_time_steps):
            if self.stop:
                logger.info("Training stopped at time %d", step)
                await self.socket_manager_service.emit("stopped", {})
                break

            (
                data_messages,
                houses_messages,
            ) = self.client_manager_service.update_data(
                obs_dict=self.obs_dict, time_step=step
            )

            await self.socket_manager_service.emit("houseChange", houses_messages)
            await self.socket_manager_service.emit("dataChange", data_messages)

            sleep(self.speed)

            # Select action with probabilities
            action_and_prob = {
                k: self.agent.select_action(
                    normStateDict(self.obs_dict[k], config_dict)
                )
                for k in self.obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in self.obs_dict.keys()}
            action_prob = {k: action_and_prob[k][1] for k in self.obs_dict.keys()}

            # Take action and get new transition
            next_obs_dict, rewards_dict = self.env._step(action)

            # Episode is done
            done = step % self.time_steps_per_episode == self.time_steps_per_episode - 1

            # Storing in replay buffer
            # TODO: remember to remove normStateDict and config dict from agents
            self.agent.store_transition(
                obs_dict=self.obs_dict,
                action=action,
                action_prob=action_prob,
                rewards_dict=rewards_dict,
                next_obs_dict=next_obs_dict,
                done=done,
            )

            # Update metrics
            self.metrics_service.update(
                self.obs_dict, next_obs_dict, rewards_dict, step
            )

            # Set next state as current state
            self.obs_dict = next_obs_dict

            # New episode, reset environment
            if done:
                logger.info("New episode at time %d", step)
                await self.socket_manager_service.emit(
                    "success", {"message": f"New episode at time {step}"}
                )

                self.obs_dict = self.env._reset()

            # Epoch: update agent
            await self.update_agent(step)

            # Log train statistics
            await self.log_train_stats(step)

            # Test policy
            self.test_policy(step)

        await self.end_simulation()

    def test(self, tr_time_steps: int) -> None:
        """
        Test ppo agent on an episode of nb_test_timesteps, with
        """
        env = deepcopy(self.env)
        cumul_avg_reward = 0.0
        cumul_temp_error = 0.0
        cumul_signal_error = 0.0
        obs_dict = env._reset()
        with torch.no_grad():
            for _ in range(self.nb_time_steps_test):
                action_and_prob = {
                    k: self.agent.select_action(normStateDict(obs_dict[k], config_dict))
                    for k in obs_dict.keys()
                }
                action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
                obs_dict, rewards_dict = env._step(action)
                for i in range(self.nb_agents):
                    cumul_avg_reward += rewards_dict[i] / self.nb_agents
                    cumul_temp_error += (
                        np.abs(obs_dict[i]["indoor_temp"] - obs_dict[i]["target_temp"])
                        / self.nb_agents
                    )
                    cumul_signal_error += np.abs(
                        obs_dict[i]["reg_signal"] - obs_dict[i]["cluster_hvac_power"]
                    ) / (self.nb_agents**2)
        mean_avg_return = cumul_avg_reward / self.nb_time_steps_test
        mean_temp_error = cumul_temp_error / self.nb_time_steps_test
        mean_signal_error = cumul_signal_error / self.nb_time_steps_test

        self.metrics_service.log_test_results(
            mean_avg_return, mean_temp_error, mean_signal_error, tr_time_steps
        )

    def test_policy(self, step: int) -> None:
        if step % self.time_steps_test_log == self.time_steps_test_log - 1:
            logger.info("Testing at time %d", step)
            self.test(step)

        if (
            self.save_actor_name
            and step % self.time_steps_per_saving_actor == 0
            and step != 0
        ):
            path = os.path.join(".", "actors", self.save_actor_name)
            self.agent.save(path, step)
            self.metrics_service.save_actor(
                os.path.join(path, "actor" + str(step) + ".pth")
            )

    async def update_agent(self, step: int) -> None:
        if step % self.time_steps_per_epoch == self.time_steps_per_epoch - 1:
            logger.info("Updating agent at time %d", step)
            await self.socket_manager_service.emit(
                "success", {"message": f"Updating agent at time {step}"}
            )

            self.agent.update(step)

    async def log_train_stats(self, step: int) -> None:
        if step % self.time_steps_train_log == self.time_steps_train_log - 1:
            logger.info("Logging stats at time %d", step)
            await self.socket_manager_service.emit(
                "success", {"message": f"Logging stats at time {step}"}
            )

            self.metrics_service.log(
                step, self.time_steps_train_log, self.env.date_time
            )
        self.metrics_service.reset()

    async def end_simulation(self) -> None:
        if self.save_actor_name:
            path = os.path.join(".", "actors", self.save_actor_name)
            self.agent.save(path)
            self.metrics_service.save_actor(os.path.join(path, "actor.pth"))

        logger.info("Simulation ended")
        await self.socket_manager_service.emit("stopped", {})
        await self.socket_manager_service.emit(
            "success", {"message": "Simulation ended"}
        )

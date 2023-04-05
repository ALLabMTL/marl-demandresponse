import os
import random
from copy import deepcopy
from time import sleep
from typing import Dict, List, Tuple

import numpy as np
from app.core.agents.trainables.trainable import Trainable
import torch

from app.core.agents.trainables.ddpg import DDPG
from app.core.agents.trainables.dqn import DDQN, DQN
from app.core.agents.trainables.mappo import MAPPO
from app.core.agents.trainables.ppo import PPO
from app.core.agents.trainables.trainable import Trainable
from app.core.environment.environment import Environment
from app.core.environment.environment_properties import EnvironmentObsDict
from app.services.experiment import Experiment
from app.services.metrics_service import Metrics
from app.services.parser_service import MarlConfig
from app.services.simulation_properties import SimulationProperties
from app.utils.logger import logger
from app.utils.norm import norm_state_dict

from .client_manager_service import ClientManagerService
from .socket_manager_service import SocketManager

agents_dict: Dict[str, Tuple[Trainable, str]] = {
    "PPO": (PPO, "PPO_prop"),
    "MAPPO": (MAPPO, "PPO_prop"),
    "DQN": (DQN, "DQN_prop"),
    "DDPG": (DDPG, "DDPG_prop"),
    "DDQN": (DDQN, "DQN_prop"),
}


class TrainingManager(Experiment):
    env: Environment
    nb_agents: int
    speed: float
    obs_dict: List[EnvironmentObsDict]
    agent: Trainable
    static_props: SimulationProperties

    @property
    def time_steps_per_episode(self) -> int:
        return int(self.static_props.nb_time_steps / self.static_props.nb_episodes)

    @property
    def time_steps_train_log(self) -> int:
        return int(self.static_props.nb_time_steps / self.static_props.nb_logs)

    @property
    def time_steps_per_epoch(self) -> int:
        return int(self.static_props.nb_time_steps / self.static_props.nb_epochs)

    @property
    def time_steps_test_log(self) -> int:
        return int(self.static_props.nb_time_steps / self.static_props.nb_test_logs)

    @property
    def time_steps_per_saving_actor(self) -> int:
        return int(
            self.static_props.nb_time_steps / self.static_props.nb_inter_saving_actor
        )

    def __init__(
        self,
        client_manager_service: ClientManagerService,
        metrics_service: Metrics,
    ) -> None:
        self.client_manager_service = client_manager_service
        self.metrics_service = metrics_service
        self.stop = False

    def initialize(self, config: MarlConfig) -> None:
        random.seed(config.simulation_props.net_seed)
        self.static_props = config.simulation_props
        logger.info("Initializing environment...")
        self.env = Environment(config.env_prop)
        self.nb_agents = config.env_prop.cluster_prop.nb_agents
        self.speed = 2.0
        self.obs_dict = self.env.reset()
        self.num_state = len(norm_state_dict(self.obs_dict, self.env.init_props)[0])
        self.metrics_service.initialize(
            self.nb_agents,
            self.static_props.start_stats_from,
            self.static_props.nb_time_steps,
        )
        self.agent = agents_dict[self.static_props.agent][0](
            config=getattr(config, agents_dict[self.static_props.agent][1]),
            num_state=self.num_state,
        )
        self.client_manager_service.initialize_data(config.CLI_config.interface)
        logger.info("Number of states: %d", self.num_state)

    async def start(self, config: MarlConfig) -> None:
        self.initialize(config)
        await self.client_manager_service.log(
            emit=True,
            endpoint="agent",
            data=self.static_props.agent,
        )
        await self.client_manager_service.log(
            text="Starting simulation...",
            emit=True,
            endpoint="success",
            data={"message": "Starting simulation"},
        )

        self.obs_dict = self.env.reset()

        for step in range(self.static_props.nb_time_steps):
            # Check if UI stopped or paused simulation
            if await self.should_stop(step):
                break

            # Update data that will be sent to UI
            await self.client_manager_service.update_data(
                obs_dict=self.obs_dict, time_step=step
            )
            # Wait time of the interface
            sleep(self.speed)

            # Select action with probabilities
            actions = self.agent.select_actions(
                norm_state_dict(self.obs_dict, self.env.init_props)
            )

            # Take action and get new transition
            next_obs_dict, rewards_dict = self.env.step(actions)

            # Episode is done
            done = step % self.time_steps_per_episode == self.time_steps_per_episode - 1

            # Storing in replay buffer
            self.agent.store_transition(
                norm_state_dict(self.obs_dict, self.env.init_props),
                norm_state_dict(next_obs_dict, self.env.init_props),
                rewards_dict,
                done,
            )

            # Update metrics
            self.metrics_service.update(
                self.obs_dict, next_obs_dict, rewards_dict, step
            )

            # Set next state as current state
            self.obs_dict = next_obs_dict

            # If new episode, reset environment
            await self.reset_environment(step)

            # Epoch: update agent
            await self.update_agent(step)

            # Log train statistics
            await self.log_statistics(step)

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
        obs_dict = env.reset()
        with torch.no_grad():
            for _ in range(self.static_props.nb_time_steps_test):
                actions = self.agent.select_actions(
                    norm_state_dict(obs_dict, self.env.init_props)
                )
                obs_dict, rewards_dict = env.step(actions)
                for i in range(self.nb_agents):
                    cumul_avg_reward += rewards_dict[i] / self.nb_agents
                    cumul_temp_error += (
                        np.abs(obs_dict[i]["indoor_temp"] - obs_dict[i]["target_temp"])
                        / self.nb_agents
                    )
                    cumul_signal_error += np.abs(
                        obs_dict[i]["reg_signal"] - obs_dict[i]["cluster_hvac_power"]
                    ) / (self.nb_agents**2)
        mean_avg_return = cumul_avg_reward / self.static_props.nb_time_steps_test
        mean_temp_error = cumul_temp_error / self.static_props.nb_time_steps_test
        mean_signal_error = cumul_signal_error / self.static_props.nb_time_steps_test

        self.metrics_service.log_test_results(
            mean_avg_return, mean_temp_error, mean_signal_error, tr_time_steps
        )

    def test_policy(self, step: int) -> None:
        if step % self.time_steps_test_log == self.time_steps_test_log - 1:
            logger.info("Testing at time %d", step)
            self.test(step)

        if (
            self.static_props.save_actor_name
            and step % self.time_steps_per_saving_actor == 0
            and step != 0
        ):
            path = os.path.join(".", "actors", self.static_props.save_actor_name)
            self.agent.save(path, step)
            self.metrics_service.save_actor(
                os.path.join(path, "actor" + str(step) + ".pth")
            )

    async def update_agent(self, step: int) -> None:
        if step % self.time_steps_per_epoch == self.time_steps_per_epoch - 1:
            await self.client_manager_service.log(
                text=f"Updating agent at time {step}",
                emit=True,
                endpoint="success",
                data={"message": f"Updating agent at time {step}"},
            )
            self.agent.update(step)

    async def log_statistics(self, step: int) -> None:
        if step % self.time_steps_train_log == self.time_steps_train_log - 1:
            await self.client_manager_service.log(
                text=f"Logging stats at time {step}",
                emit=True,
                endpoint="success",
                data={"message": f"Logging stats at time {step}"},
            )
            self.metrics_service.log(
                step, self.time_steps_train_log, self.env.date_time
            )
            self.metrics_service.reset()

    async def end_simulation(self) -> None:
        if self.static_props.save_actor_name:
            path = os.path.join(".", "actors", self.static_props.save_actor_name)
            self.agent.save(path)
            self.metrics_service.save_actor(os.path.join(path, "actor.pth"))

        await self.client_manager_service.log(emit=True, endpoint="stopped", data={})
        await self.client_manager_service.log(
            emit=True,
            endpoint="success",
            text="Simulation ended",
            data={"message": "Simulation ended"},
        )

    async def should_stop(self, step) -> bool:
        if self.stop:
            await self.client_manager_service.log(
                endpoint="stopped", data={}, text=f"Training stopped at time {step}"
            )
            return True
        return False

    async def reset_environment(self, step) -> None:
        if step % self.time_steps_per_episode == self.time_steps_per_episode - 1:
            await self.client_manager_service.log(
                text=f"New episode at time {step}",
                emit=True,
                endpoint="success",
                data={"message": f"New episode at time {step}"},
            )
            self.obs_dict = self.env.reset()

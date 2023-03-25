import random
from datetime import datetime
from time import sleep
from typing import Dict, List, Union

from app.config import config_dict
from app.core.environment.environment import Environment
from app.utils.logger import logger
from app.services.metrics_service import Metrics
from app.utils.utils import normStateDict
from app.core.agents.controllers.controller import Controller
from app.services.experiment import Experiment
from app.core.agents.controllers import (
    AlwaysOnController,
    BangBangController,
    BasicController,
    DDPGAgent,
    DeadbandBangBangController,
    DQNAgent,
    GreedyMyopic,
    MPCController,
    PPOAgent,
)

from .client_manager_service import ClientManagerService
from .socket_manager_service import SocketManager


agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "PPO": PPOAgent,
    "MAPPO": PPOAgent,
    "DQN": DQNAgent,
    "GreedyMyopic": GreedyMyopic,
    "MPC": MPCController,
    "MADDPG": DDPGAgent,
}


class ControllerManager(Experiment):
    env: Environment
    nb_agents: int
    obs_dict: Dict[int, List[Union[float, str, bool, datetime]]]
    nb_episodes: int
    nb_time_steps: int
    nb_test_logs: int
    nb_logs: int
    start_stats_from: int
    num_state: int
    actor_name: str
    agent: str
    net_seed: int
    time_steps_per_episode: int
    time_steps_train_log: int
    time_steps_test_log: int
    actors: Dict[int, Controller]
    stop: bool
    pause: bool
    speed: float

    def __init__(
        self,
        socket_manager_service: SocketManager,
        client_manager_service: ClientManagerService,
        metrics_service: Metrics,
    ) -> None:
        self.client_manager_service = client_manager_service
        self.socket_manager_service = socket_manager_service
        self.metrics_service = metrics_service
        self.stop = False
        self.pause = False

    def initialize(self) -> None:
        random.seed(1)
        self.env = Environment()
        self.nb_agents = self.env.cluster.nb_agents
        # TODO: Get these arguments from config file (parser)
        self.nb_episodes = 3
        self.nb_time_steps = 1000
        self.nb_test_logs = 100
        self.nb_logs = 100
        self.actor_name: str = "PPO"
        self.net_seed: int = 4
        self.agent_name: str = "BangBang"
        self.start_stats_from = 0
        self.speed = 2.0
        self.obs_dict = self.env._reset()
        self.num_state = len(
            normStateDict(self.obs_dict[next(iter(self.obs_dict))], config_dict)
        )

        self.metrics_service.initialize(
            self.nb_agents, self.start_stats_from, self.nb_time_steps
        )

        # TODO: Get agent from config file
        self.actors: Dict[int, Controller] = {}

        for house_id in range(self.nb_agents):
            agent_prop: Dict[str, Union[str, int]] = {"id": house_id}
            # TODO: Change how we init agents (not all configdict is necessary, only agent props)
            if self.actor_name:
                agent_prop.update(
                    {"actor_name": self.actor_name, "net_seed": self.net_seed}
                )

            self.actors[house_id] = agents_dict[self.agent_name](
                agent_prop, config_dict, num_state=self.num_state
            )

        self.time_steps_per_episode = int(self.nb_time_steps / self.nb_episodes)
        self.time_steps_train_log = int(self.nb_time_steps / self.nb_logs)
        self.time_steps_test_log = int(self.nb_time_steps / self.nb_test_logs)
        logger.info("Number of states: %d", self.num_state)

    async def start(self) -> None:
        # Initialize training variables
        logger.info("Initializing environment...")
        await self.socket_manager_service.emit(
            "success", {"message": "Initializing environment..."}
        )

        self.initialize()
        self.client_manager_service.initialize_data()
        await self.socket_manager_service.emit("agent", self.agent_name)
        await self.socket_manager_service.emit(
            "success", {"message": "Starting simulation"}
        )
        self.obs_dict = self.env._reset()

        for step in range(self.nb_time_steps):
            if self.pause:
                await self.socket_manager_service.emit("paused", {})
                logger.debug("simulation paused")
                while(True):
                    if(not self.pause or self.stop):
                        break
            
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

            # Take action and get new transition
            actions = self.get_actions(self.obs_dict)
            next_obs_dict, rewards_dict = self.env._step(actions)

            # Update metrics
            self.metrics_service.update(
                self.obs_dict, next_obs_dict, rewards_dict, step
            )

            # Set next state as current state
            self.obs_dict = next_obs_dict

            # If new episode, reset environment
            if step % self.time_steps_per_episode == self.time_steps_per_episode - 1:
                logger.info("New episode at time %d", step)
                await self.socket_manager_service.emit(
                    "success", {"message": f"New episode at time {step}"}
                )
                self.obs_dict = self.env._reset()

            # Log train statistics
            if step % self.time_steps_train_log == self.time_steps_train_log - 1:
                logger.info("Logging stats at time %d", step)
                await self.socket_manager_service.emit(
                    "success", {"message": f"Logging stats at time {step}"}
                )

                self.metrics_service.log(
                    step, self.time_steps_train_log, self.env.date_time
                )
                self.metrics_service.reset()

        logger.info("Simulation ended")
        self.metrics_service.update_final()
        await self.socket_manager_service.emit("stopped", {})
        await self.socket_manager_service.emit(
            "success", {"message": "Simulation ended"}
        )

    def get_actions(self, obs_dict) -> dict:
        actions = {}
        for agent_id, actor in self.actors.items():
            actions[agent_id] = actor.act(obs_dict)
        return actions

import datetime
import random
from time import sleep
from typing import Dict, List, Union

from app.config import config_dict
from app.core.agents.controllers.bangbang_controllers import (
    AlwaysOnController,
    BangBangController,
    BasicController,
    DeadbandBangBangController,
)
from app.core.agents.controllers.controller import Controller
from app.core.agents.controllers.greedy_myopic_controller import GreedyMyopic
from app.core.agents.controllers.mpc_controller import MPCController
from app.core.agents.controllers.rl_controllers import (
    DDPGController,
    DQNController,
    PPOController,
)
from app.core.environment.environment import Environment
from app.services.experiment import Experiment
from app.services.metrics_service import Metrics
from app.services.parser_service import ParserService
from app.utils.logger import logger
from app.utils.utils import normStateDict

from .client_manager_service import ClientManagerService
from .socket_manager_service import SocketManager

agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "PPO": PPOController,
    "MAPPO": PPOController,
    "DQN": DQNController,
    "GreedyMyopic": GreedyMyopic,
    "MPC": MPCController,
    "DDPG": DDPGController,
}


class ControllerManager(Experiment):
    env: Environment
    nb_agents: int
    obs_dict: Dict[int, List[Union[float, str, bool, datetime.datetime]]]
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
        parser_service: ParserService,
    ) -> None:
        self.client_manager_service = client_manager_service
        self.socket_manager_service = socket_manager_service
        self.metrics_service = metrics_service
        self.stop = False
        self.static_props = parser_service.config.controller_props
        self.parser_service = parser_service

    def initialize(self) -> None:
        random.seed(1)
        self.env = Environment(self.parser_service.config.env_prop)
        self.nb_agents = self.env.cluster.nb_agents
        self.agent_name: str = self.parser_service.config.controller_props.agent
        self.nb_time_steps = self.parser_service.config.controller_props.nb_time_steps
        self.pause = False
        self.speed = 2.0
        self.obs_dict = self.env._reset()
        self.num_state = len(
            normStateDict(self.obs_dict[next(iter(self.obs_dict))], config_dict)
        )

        self.metrics_service.initialize(
            self.nb_agents,
            self.static_props.start_stats_from,
            self.static_props.nb_time_steps,
        )

        # TODO: Get agent from config file
        self.actors: Dict[int, Controller] = {}

        for house_id in range(self.nb_agents):
            agent_prop: Dict[str, Union[str, int]] = {"id": house_id}
            # TODO: Change how we init agents (not all configdict is necessary, only agent props)
            if self.static_props.actor_name:
                agent_prop.update(
                    {
                        "actor_name": self.static_props.actor_name,
                        "net_seed": self.static_props.net_seed,
                    }
                )

            self.actors[house_id] = agents_dict[self.static_props.agent](
                agent_prop, config_dict, num_state=self.num_state
            )

        self.time_steps_per_episode = int(
            self.static_props.nb_time_steps / self.static_props.nb_episodes
        )
        self.time_steps_train_log = int(
            self.static_props.nb_time_steps / self.static_props.nb_logs
        )
        self.time_steps_test_log = int(
            self.static_props.nb_time_steps / self.static_props.nb_test_logs
        )
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
                logger.debug("simulation paused")
                await self.socket_manager_service.emit("paused", {})
                await self.socket_manager_service.sleep(0)
                while(True):
                    if(not self.pause or self.stop):
                        await self.socket_manager_service.sleep(0)
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

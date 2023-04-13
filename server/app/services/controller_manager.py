import random
from time import sleep
from typing import Dict, List, Union

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
from app.core.environment.environment_properties import EnvironmentObsDict
from app.services.experiment import Experiment
from app.services.metrics_service import Metrics
from app.services.parser_service import MarlConfig
from app.services.simulation_properties import SimulationProperties
from app.services.socket_manager_service import SocketManager
from app.utils.logger import logger
from app.utils.norm import norm_state_dict

from .client_manager_service import ClientManagerService

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
    speed: float
    obs_dict: List[EnvironmentObsDict]
    num_state: int
    actors: Dict[int, Controller]
    static_props: SimulationProperties
    current_time_step: int

    @property
    def time_steps_per_episode(self) -> int:
        return int(self.static_props.nb_time_steps / self.static_props.nb_episodes)

    @property
    def time_steps_train_log(self) -> int:
        return int(self.static_props.nb_time_steps / self.static_props.nb_logs)

    def __init__(
        self,
        client_manager_service: ClientManagerService,
        socket_manager_service: SocketManager,
        metrics_service: Metrics,
    ) -> None:
        self.client_manager_service = client_manager_service
        self.socket_manager_service = socket_manager_service
        self.metrics_service = metrics_service
        self.stop = False
        self.pause = False

    async def initialize(self, config: MarlConfig) -> None:
        random.seed(config.simulation_props.net_seed)
        self.current_time_step = 0
        self.stop = False
        self.static_props = config.simulation_props
        logger.info("Initializing environment...")
        self.env = Environment(config.env_prop)
        self.nb_agents = config.env_prop.cluster_prop.nb_agents
        self.speed = 2.0
        self.obs_dict = self.env.reset()
        self.num_state = len(norm_state_dict(self.obs_dict, self.env.init_props)[0])
        self.metrics_service.initialize(
            self.nb_agents,
            config.simulation_props.start_stats_from,
            config.simulation_props.nb_time_steps,
        )
        self.actors: Dict[int, Controller] = {}
        self.initialize_actors(config)
        self.client_manager_service.initialize_data(config.CLI_config.interface)
        logger.info("Number of states: %d", self.num_state)
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

    async def start(self, config: MarlConfig) -> None:
        if not self.pause or self.stop:
            await self.initialize(config)
        else: 
            await self.client_manager_service.log(
                text=f"Continuing simulation",
                emit=True,
                endpoint="success",
                data={"message": f"Continuing simulation"},
            )
            
        self.pause = False
        
        for step in range(self.current_time_step, self.static_props.nb_time_steps):
            # Check if UI stopped or paused simulation
            if await self.should_stop(step):
                break
            # Update data that will be sent to UI
            await self.client_manager_service.update_data(
                obs_dict=self.obs_dict, time_step=step
            )
            # Wait time of the interface
            sleep(self.speed)

            # Take action and get new transition
            actions = self.get_actions(self.obs_dict)
            next_obs_dict, rewards_dict = self.env.step(actions)

            # Update metrics
            self.metrics_service.update(
                self.obs_dict, next_obs_dict, rewards_dict, step
            )

            # Set next state as current state
            self.obs_dict = next_obs_dict

            # If new episode, reset environment
            await self.reset_environment(step)

            # Log train statistics
            await self.log_statistics(step)
            self.current_time_step +=1

        await self.emit_stop()


    async def emit_stop(self):
        if self.stop or (self.current_time_step == self.static_props.nb_time_steps):
            self.metrics_service.update_final()
            await self.client_manager_service.log(emit=True, endpoint="stopped", data={})
            await self.client_manager_service.log(
                emit=True,
                endpoint="success",
                text="Simulation ended",
                data={"message": "Simulation ended"},
            )

    def get_actions(self, obs_dict) -> dict:
        actions = {}
        for agent_id, actor in self.actors.items():
            actions[agent_id] = actor.act(obs_dict)
        return actions

    async def reset_environment(self, step) -> None:
        if step % self.time_steps_per_episode == self.time_steps_per_episode - 1:
            await self.client_manager_service.log(
                text=f"New episode at time {step}",
                emit=True,
                endpoint="success",
                data={"message": f"New episode at time {step}"},
            )
            self.obs_dict = self.env.reset()

    async def log_statistics(self, step) -> None:
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

    async def should_stop(self, step) -> bool:
        if self.stop:
            await self.client_manager_service.log(
                emit = True, endpoint="stopped", data={}, text=f"Simulation stopped at time {step}"
            )
            return True
        elif self.pause:
            await self.client_manager_service.log(
                emit= True, endpoint="paused", data={}, text=f"Simulation paused at time {step}"
            )
            return True
        
        return False

    def initialize_actors(self, config: MarlConfig) -> None:
        for house_id in range(self.nb_agents):
            agent_prop: Dict[str, Union[str, int]] = {"id": house_id}
            if config.simulation_props.agent:
                agent_prop.update(
                    {
                        "actor_name": config.simulation_props.agent,
                        "net_seed": config.simulation_props.net_seed,
                    }
                )

            self.actors[house_id] = agents_dict[config.simulation_props.agent](
                agent_prop, config, num_state=self.num_state
            )

    async def stop_sim(self, stop_state: bool) -> None:
        if stop_state:
            await self.end_simulation()

        self.stop = stop_state

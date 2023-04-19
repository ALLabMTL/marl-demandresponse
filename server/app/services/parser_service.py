from typing import Union

from pydantic import BaseModel, Field

from app.core.agents.trainables.ddpg import DDPGProperties
from app.core.agents.trainables.dqn import DQNProperties
from app.core.agents.trainables.mappo import MAPPOProperties
from app.core.agents.trainables.ppo import PPOProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.services.simulation_properties import SimulationProperties
from app.utils.logger import logger


class MPCProperties(BaseModel):
    """
    Properties for MPC agent.

    This class represents the properties for the MPC agent.

    Attributes:
        rolling_horizon (int): Rolling horizon for the MPC agent.
    """

    rolling_horizon: int = Field(
        default=15,
        description="Rolling horizon for the MPC agent.",
    )


class CLIConfig(BaseModel):
    """
    Properties ported from the CIL calls.

    This class represents the properties ported from the CIL calls.

    Attributes:
        experiment_name (str): Name of the experiment.
        interface (bool): Whether we want to launch the interface with the simulation.
        wandb (bool): Whether to use wandb.
    """

    experiment_name: str = Field(
        default="default",
        description="Name of the experiment.",
    )
    interface: bool = Field(
        default=True,
        description="Whether we want to launch the interface with the simulation",
    )
    wandb: bool = Field(
        default=False,
        description="Whether to use wandb.",
    )
    wandb_project: str = Field(
        default="myproject",
        description="Project name for wandb.",
    )


class MarlConfig(BaseModel):
    """
    Configuration for the multi-agent reinforcement learning (MARL) environment.

    This class represents the configuration data for the MARL environment. It contains properties
    for the CLI, simulation, environment, PPO, MAPPO, DDPG, DQN, and MPC agents.

    Attributes:
        CLI_config (CLIConfig): Properties ported from the CIL calls.
        simulation_props (SimulationProperties): Properties for the simulation.
        env_prop (EnvironmentProperties): Properties for the environment.
        PPO_prop (PPOProperties): Properties for the PPO agent.
        MAPPO_prop (MAPPOProperties): Properties for the MAPPO agent.
        DDPG_prop (DDPGProperties): Properties for the DDPG agent.
        DQN_prop (DQNProperties): Properties for the DQN agent.
        MPC_prop (MPCProperties): Properties for the MPC agent.
    """

    CLI_config: CLIConfig = CLIConfig()
    simulation_props: SimulationProperties = SimulationProperties()
    env_prop: EnvironmentProperties = EnvironmentProperties()
    PPO_prop: PPOProperties = PPOProperties()
    MAPPO_prop: MAPPOProperties = MAPPOProperties()
    DDPG_prop: DDPGProperties = DDPGProperties()
    DQN_prop: DQNProperties = DQNProperties()
    MPC_prop: MPCProperties = MPCProperties()


class ParserService:
    """
    Service for parsing configuration data.

    This class represents the service for parsing configuration data. It reads the configuration
    data from a JSON file and creates a MarlConfig object.

    Attributes:
        config (MarlConfig): Configuration data for the MARL environment.
    """

    def __init__(self) -> None:
        """
        Initialize the ParserService object.

        This method creates a new ParserService object and loads the configuration data for the MARL environment.
        If the configuration file is not found, it uses the default configuration.
        """
        try:
            self.config = MarlConfig.parse_file("core/config/MARLconfig.json")
        except FileNotFoundError:
            logger.warning("MARLconfig.json not found, using default config.")
            self.config = MarlConfig()

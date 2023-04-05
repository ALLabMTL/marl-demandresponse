from pydantic import BaseModel, Field

from app.core.agents.trainables.ddpg import DDPGProperties
from app.core.agents.trainables.dqn import DQNProperties
from app.core.agents.trainables.mappo import MAPPOProperties
from app.core.agents.trainables.ppo import PPOProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.services.simulation_properties import SimulationProperties
from app.utils.logger import logger


class MPCProperties(BaseModel):
    """Properties for MPC agent."""

    rolling_horizon: int = Field(
        default=15,
        description="Rolling horizon for the MPC agent.",
    )


class CLIConfig(BaseModel):
    """Properties ported from the CIL calls."""

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


class MarlConfig(BaseModel):
    """Configuration for MARL environment."""

    CLI_config: CLIConfig = CLIConfig()
    simulation_props: SimulationProperties = SimulationProperties()
    env_prop: EnvironmentProperties = EnvironmentProperties()
    PPO_prop: PPOProperties = PPOProperties()
    MAPPO_prop: MAPPOProperties = MAPPOProperties()
    DDPG_prop: DDPGProperties = DDPGProperties()
    DQN_prop: DQNProperties = DQNProperties()
    MPC_prop: MPCProperties = MPCProperties()


class ParserService:
    def __init__(self) -> None:
        try:
            self.config = MarlConfig.parse_file("core/config/MARLconfig.json")
        except FileNotFoundError:
            logger.warning("MARLconfig.json not found, using default config.")
            self.config = MarlConfig()

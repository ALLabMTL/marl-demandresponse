from typing import Literal

from pydantic import BaseModel, Field

from app.core.agents.ddpg import DDPGProperties
from app.core.agents.dqn import DQNProperties
from app.core.agents.mappo import MAPPOProperties
from app.core.agents.ppo import PPOProperties
from app.core.environment.cluster.building_properties import (
    BuildingNoiseProperties,
    BuildingProperties,
)
from app.core.environment.cluster.hvac import HvacNoiseProperties, HvacProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.services.controller_propreties import ControllerPropreties


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
    wandb: bool = Field(
        default=False,
        description="Whether to use wandb.",
    )
    log_metrics_path: str = Field(
        default="",
        description="Path to the metrics file.",
    )
    nb_time_steps: int = Field(
        default=100000,
        description="Number of time steps for the experiment.",
    )
    save_actor_name: str = Field(
        default="",
        description="Name of the actor to save.",
    )
    nb_inter_saving_actor: int = Field(
        default=0,
        description="Number of time steps between saving the actor.",
    )
    mode: Literal["train", "simulation"] = Field(
        default="simulation",
        description="Mode of the experiment.",
    )


class MarlConfig(BaseModel):
    """Configuration for MARL environment."""

    house_prop: BuildingProperties = BuildingProperties()
    noise_house_prop: BuildingNoiseProperties = BuildingNoiseProperties()
    hvac_prop: HvacProperties = HvacProperties()
    noise_hvac_prop: HvacNoiseProperties = HvacNoiseProperties()
    env_prop: EnvironmentProperties = EnvironmentProperties()
    PPO_prop: PPOProperties = PPOProperties()
    MAPPO_prop: MAPPOProperties = MAPPOProperties()
    DDPG_prop: DDPGProperties = DDPGProperties()
    DQN_prop: DQNProperties = DQNProperties()
    MPC_prop: MPCProperties = MPCProperties()
    CLI_config: CLIConfig = CLIConfig()
    controller_props: ControllerPropreties = ControllerPropreties()


class ParserService:
    def __init__(self) -> None:
        pass

    config: MarlConfig = MarlConfig()

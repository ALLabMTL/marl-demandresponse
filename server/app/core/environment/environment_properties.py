import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.core.environment.cluster.cluster_properties import (
    AgentsCommunicationProperties,
    TemperatureProperties,
)
from app.core.environment.cluster.building_properties import BuildingProperties
from app.core.environment.power_grid.power_grid_properties import (
    PowerGridProperties,
)


class PenaltyProperties(BaseModel):
    mode: Literal["common_L2", "individual_L2", "common_max_error", "mixture"] = Field(
        default="individual_L2", description="Mode of temperature penalty"
    )
    alpha_ind_l2: float = 1.0
    alpha_common_l2: float = 1.0
    alpha_common_max: float = 0.0


class RewardProperties(BaseModel):
    """Properties of the reward function."""

    alpha_temp: float = Field(
        default=1.0,
        description="Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    alpha_sig: float = Field(
        default=1.0,
        description="Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    norm_reg_sig: int = Field(
        default=7500,
        description="Average power use, for signal normalization.",
        alias="norm_reg_sig",
    )
    penalty_props: PenaltyProperties = PenaltyProperties()
    sig_penalty_mode: Literal["common_L2"] = Field(
        default="common_L2",
        description="Mode of signal penalty (only common_L2 available)",
    )


class StateProperties(BaseModel):
    """Properties of the state space."""

    hour: bool = Field(
        default=False,
        description="Whether to include the hour of the day in the state space.",
    )
    day: bool = Field(
        default=False,
        description="Whether to include the day of the week in the state space.",
    )
    solar_gain: bool = Field(
        default=False,
        description="Whether to include solar gain in the state space.",
    )
    thermal: bool = Field(
        default=False,
        description="Whether to include thermal state in the state space.",
    )
    hvac: bool = Field(
        default=False,
        description="Whether to include hvac state in the state space.",
    )


class Neighbours2D(BaseModel):
    row_size: int = Field(
        default=5,
        description="Number of rows in the 2D grid of neighbours.",
    )
    distance_comm: int = Field(
        default=2,
        description="Distance of communication between neighbours.",
    )


class MessageProperties(BaseModel):
    """Properties of the message space."""

    thermal: bool = Field(
        default=False,
        description="Whether to include thermal state in the message space.",
    )
    hvac: bool = Field(
        default=False,
        description="Whether to include hvac state in the message space.",
    )


class ClusterPropreties(BaseModel):
    nb_agents: int = Field(
        default=10,
        description="Number of agents in the cluster.",
    )
    nb_agents_comm: int = Field(
        default=10,
        description="Maximal number of houses a single house communicates with.",
    )
    # TODO: make field
    agents_comm_prop: AgentsCommunicationProperties = AgentsCommunicationProperties()
    house_prop: BuildingProperties = BuildingProperties()
    message_prop: MessageProperties = MessageProperties()


class EnvironmentProperties(BaseModel):
    """Properties of the environment."""

    start_datetime: datetime.datetime = Field(
        default=datetime.datetime(2021, 1, 1, 0, 0, 0),
        description="Start date and time (Y-m-d H:M:S).",
    )
    start_datetime_mode: Literal["fixed", "random"] = Field(
        default="fixed",
        description="Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)",
    )
    time_step: datetime.timedelta = Field(
        default=datetime.timedelta(0, 4),
        description="How long a timestep should take (in seconds).",
    )

    temp_prop: TemperatureProperties = TemperatureProperties()
    state_prop: StateProperties = StateProperties()
    reward_prop: RewardProperties = RewardProperties()
    cluster_prop: ClusterPropreties = ClusterPropreties()
    power_grid_prop: PowerGridProperties = PowerGridProperties()

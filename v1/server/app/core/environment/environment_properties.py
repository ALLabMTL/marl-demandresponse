import datetime
from typing import Dict
from pydantic import BaseModel, Field


class PenaltyProperties(BaseModel):
    mode: str = "common_L2"
    alpha_ind_l2: float = 1.0
    alpha_common_l2: float = 1.0
    alpha_common_max: float = 0.0


class RewardProperties(BaseModel):
    """Properties of the reward function."""

    alpha_temp: float = Field(
        1.0,
        description="Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    alpha_sig: float = Field(
        1.0,
        description="Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    norm_reg_signal: int = Field(
        7500,
        description="Average power use, for signal normalization.",
    )
    penalty_props: PenaltyProperties = PenaltyProperties()


class StateProperties(BaseModel):
    """Properties of the state space."""

    hour: bool = Field(
        False,
        description="Whether to include the hour of the day in the state space.",
    )
    day: bool = Field(
        False,
        description="Whether to include the day of the week in the state space.",
    )
    solar_gain: bool = Field(
        False,
        description="Whether to include solar gain in the state space.",
    )
    thermal: bool = Field(
        False,
        description="Whether to include thermal state in the state space.",
    )
    hvac: bool = Field(
        False,
        description="Whether to include hvac state in the state space.",
    )


class EnvironmentProperties(BaseModel):
    """Properties of the environment."""

    start_datetime: datetime.datetime = datetime.datetime(
        2021, 1, 1, 0, 0, 0
    )  # Start date and time (Y-m-d H:M:S)
    # TODO: random or fixed validator
    start_datetime_mode: str = "fixed"  # Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)
    time_step: datetime.timedelta = datetime.timedelta(0, 4)
    state_properties: StateProperties = StateProperties()
    message_properties: Dict[str, bool] = {"thermal": False, "hvac": False}
    reward_properties: RewardProperties = RewardProperties()

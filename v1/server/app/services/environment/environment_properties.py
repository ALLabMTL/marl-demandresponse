import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel
from v1.server.app.services.environment.cluster.cluster_properties import (
    ClusterProperties,
)
from v1.server.app.services.environment.power_grid.power_grid_properties import (
    PowerGridProperties,
)

StateProperty: Enum("StateProperty", ["hour", "day", "solar_gain", "thermal", "hvac"])
PenaltyMode = Enum("PenaltyMode", ["individual_l2", "common_l2", "common_max_error"])
DateTimeMode = Enum("DateTimeMode", ["random", "fixed"])
MessageProperty: Enum("MessageProperty", ["thermal", "hvac"])


class RewardProperties(BaseModel):
    alpha_temp: float  # Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
    alpha_sig: float  # Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
    avg_power_signal_normalization: int  # Average power use, for signal normalization
    temperature_penalty_mode: List[PenaltyMode]  # Mode of temperature penalty
    signal_penalty_mode: List[PenaltyMode]


class EnvironmentProperties(BaseModel):
    start_datetime: datetime  # Start date and time (Y-m-d H:M:S)
    start_datetime_mode: DateTimeMode  # Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)
    time_step: int
    building_cluster: ClusterProperties
    state_properties: StateProperty
    message_properties: List[MessageProperty]
    power_grid_properties: PowerGridProperties()
    reward_properties: RewardProperties()

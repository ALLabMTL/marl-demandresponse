import datetime
from typing import Dict
from pydantic import BaseModel


class PenaltyProperties(BaseModel):
    mode: str = "common_L2"
    alpha_ind_l2: float = 1.0
    alpha_common_l2: float = 1.0
    alpha_common_max: float = 0.0

class RewardProperties(BaseModel):
    alpha_temp: float = 1.0  # Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
    alpha_sig: float = 1.0  # Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
    norm_reg_signal: int = 7500  # Average power use, for signal normalization
    penalty_props: PenaltyProperties = PenaltyProperties()


class EnvironmentProperties(BaseModel):
    start_datetime: datetime.datetime = datetime.datetime(2021, 1, 1, 0, 0, 0)  # Start date and time (Y-m-d H:M:S)
    # TODO: random or fixed validator
    start_datetime_mode: str = "fixed"  # Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)
    time_step: datetime.timedelta = datetime.timedelta(0, 4)
    state_properties: dict = {
        "hour": False,
        "day": False,
        "solar_gain": False,
        "thermal": False,
        "hvac": False
    }
    message_properties: Dict[str, bool] = {"thermal": False, "hvac": False}
    reward_properties: RewardProperties = RewardProperties()

from typing import List
from pydantic import BaseModel


class HvacNoiseProperties(BaseModel):
    mode: str = "small_noise"
    # Std Gaussian noise on latent_cooling_fraction
    std_latent_cooling_fraction: float = 0.05
    # Lowest random factor for COP
    factor_COP_low: float = 0.95
    # Highest random factor for COP
    factor_COP_high: float = 1.05
    # Lowest random factor for cooling_capacity
    factor_cooling_capacity_low: float = 0.9
    # Highest random factor for cooling_capacity
    factor_cooling_capacity_high: float = 1.1
    lockout_noise: int = 0
    cooling_capacity_list: List[int] = [12500, 15000, 17500]


class HvacProperties(BaseModel):
    # TODO: check __init__ in hvac class in MADemandResponse for validations
    # coefficient of performance (power spent vs heat displaced)
    cop: float = 2.5
    # cooling capacity (W)
    cooling_capacity: int = 15000
    # fraction of latent cooling w.r.t. sensible cooling
    latent_cooling_fraction: float = 0.35
    # in seconds
    lockout_duration: int = 40

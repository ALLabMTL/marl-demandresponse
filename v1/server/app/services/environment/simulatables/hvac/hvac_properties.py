from pydantic import BaseModel


class HvacNoiseProperties(BaseModel):
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


class HvacProperties(BaseModel):
    id: int = 1
    # coefficient of performance (power spent vs heat displaced)
    cop: float = 2.5
    # cooling capacity (W)
    cooling_capacity: float = 1500.0
    # craction of latent cooling w.r.t. sensible cooling
    latent_cooling_fraction: float = 0.35
    # in seconds
    lockout_duration: int = 40
    # in seconds
    lockout_noise: int = 0
    # TODO: faire validation avec templates
    noise_mode: str = f"small_noise"

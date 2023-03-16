from typing import List

from pydantic import BaseModel, Field


class HvacNoiseProperties(BaseModel):
    std_latent_cooling_fraction: float = Field(
        default=0.05,
        description="Standard deviation of the latent cooling fraction of the HVAC.",
    )
    factor_COP_low: float = Field(
        default=0.95,
        description="Lowest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_COP_high: float = Field(
        default=1.05,
        description="Highest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_cooling_capacity_low: float = Field(
        default=0.9,
        description="Lowest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    factor_cooling_capacity_high: float = Field(
        default=1.1,
        description="Highest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    lockout_noise: int = Field(
        default=0,
        description="Lockout noise to add to the lockout duration of the HVAC.",  # TODO check if this is correct
    )
    cooling_capacity_list: List[int] = Field(
        default=[12500, 15000, 17500],
        description="List of cooling capacities to choose from randomly.",  # TODO check if this is correct
    )


class HvacProperties(
    BaseModel,
):
    cop: float = Field(
        default=2.5,
        description="coefficient of performance (ratio between cooling capacity and electric power consumption).",
        gt=0,
    )
    cooling_capacity: float = Field(
        default=15000,
        description='Rate of "negative" heat transfer produced by the HVAC (W).',
        gt=0,
    )
    latent_cooling_fraction: float = Field(
        default=0.35,
        description="Float between 0 and 1, fraction of sensible cooling (temperature) which is latent cooling (humidity).",
        gt=0,
        lt=1,
    )
    lockout_duration: int = Field(
        default=40,
        description="Duration of lockout (hardware constraint preventing to turn on the HVAC for some time after turning off), in seconds",
    )

    @property
    def max_consumption(self) -> float:
        return self.cooling_capacity / self.cop

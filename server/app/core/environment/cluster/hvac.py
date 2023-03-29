import random
from datetime import timedelta
from typing import Dict, List

import pydantic

from app.core.environment.simulatable import Simulatable


class HvacNoiseProperties(pydantic.BaseModel):
    std_latent_cooling_fraction: float = pydantic.Field(
        default=0.05,
        description="Standard deviation of the latent cooling fraction of the HVAC.",
    )
    factor_COP_low: float = pydantic.Field(
        default=0.95,
        description="Lowest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_COP_high: float = pydantic.Field(
        default=1.05,
        description="Highest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_cooling_capacity_low: float = pydantic.Field(
        default=0.9,
        description="Lowest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    factor_cooling_capacity_high: float = pydantic.Field(
        default=1.1,
        description="Highest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    lockout_noise: int = pydantic.Field(
        default=0,
        description="Lockout noise to add to the lockout duration of the HVAC.",  # TODO check if this is correct
    )
    cooling_capacity_list: List[int] = pydantic.Field(
        default=[12500, 15000, 17500],
        description="List of cooling capacities to choose from randomly.",  # TODO check if this is correct
    )


class HvacProperties(
    pydantic.BaseModel,
):
    cop: float = pydantic.Field(
        default=2.5,
        description="coefficient of performance (ratio between cooling capacity and electric power consumption).",
        gt=0,
    )
    cooling_capacity: float = pydantic.Field(
        default=15000,
        description='Rate of "negative" heat transfer produced by the HVAC (W).',
        gt=0,
    )
    latent_cooling_fraction: float = pydantic.Field(
        default=0.35,
        description="Float between 0 and 1, fraction of sensible cooling (temperature) which is latent cooling (humidity).",
        gt=0,
        lt=1,
    )
    lockout_duration: int = pydantic.Field(
        default=40,
        description="Duration of lockout (hardware constraint preventing to turn on the HVAC for some time after turning off), in seconds",
    )

    @property
    def max_consumption(self) -> float:
        return self.cooling_capacity / self.cop


class HVACObservation(HvacProperties):
    """Observation dictionary for HVAC."""

    turned_on: bool
    seconds_since_off: float
    lockout: bool
    cop: float


class HVAC(Simulatable):
    init_props: HvacProperties
    noise_props: HvacNoiseProperties
    turned_on: bool
    seconds_since_off: int
    max_consumption: int
    lockout: bool

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> dict:
        # TODO: get properties from parser_service
        self.init_props = HvacProperties()
        self.noise_props = HvacNoiseProperties()
        # TODO: put this command in HvacProperties dataclass
        self.max_consumption = self.init_props.cooling_capacity / self.init_props.cop
        self.seconds_since_off = 0
        self.lockout = False
        self.turned_on = True

        return self._get_obs()

    def _step(self, action: bool, time_step: timedelta) -> dict:
        """
        Take a step in time for the HVAC, given action of the TCL agent.

        Return:
        -

        Parameters:
        self
        action: bool, action of the TCL agent (True: ON, False: OFF)
        """
        # TODO: Make it prettier
        if not self.turned_on:
            self.seconds_since_off += time_step.seconds

        if self.turned_on or self.seconds_since_off >= self.init_props.lockout_duration:
            self.lockout = False
        else:
            self.lockout = True

        if self.lockout:
            self.turned_on = False
        else:
            self.turned_on = action
            if self.turned_on:
                self.seconds_since_off = 0
            elif (
                self.seconds_since_off + time_step.seconds
                < self.init_props.lockout_duration
            ):
                self.lockout = True
        return self._get_obs()

    def apply_noise(self) -> None:
        self.init_props.cooling_capacity = random.choices(
            self.noise_props.cooling_capacity_list
        )[0]

    def _get_obs(self) -> dict:
        obs_dict = self.init_props.dict()
        obs_dict.update(
            {
                "turned_on": self.turned_on,
                "seconds_since_off": self.seconds_since_off,
                "lockout": self.lockout,
            }
        )
        return obs_dict

    def get_Q(self) -> float:
        """
        Compute the rate of heat transfer produced by the HVAC

        Return:
        q_hvac: float, heat of transfer produced by the HVAC, in Watts

        Parameters:
        self
        """
        if self.turned_on:
            q_hvac = (
                -1
                * self.init_props.cooling_capacity
                / (1 + self.init_props.latent_cooling_fraction)
            )
        else:
            q_hvac = 0

        return q_hvac

    def get_power_consumption(self) -> int:
        """
        Compute the electric power consumption of the HVAC

        Return:
        power_cons: float, electric power consumption of the HVAC, in Watts
        """
        if self.turned_on:
            power_cons = self.max_consumption
        else:
            power_cons = 0

        return power_cons

    def message(self, hvac: bool) -> Dict[str, int]:
        message = {
            "seconds_since_off": self.seconds_since_off,
            "curr_consumption": self.get_power_consumption(),
            "max_consumption": self.max_consumption,
            "lockout_duration": self.init_props.lockout_duration,
        }
        if hvac:
            message.update(self.initial_properties.dict(exclude={"lockout_duration"}))
        return message

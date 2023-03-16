import random
from datetime import timedelta

from app.core.environment.simulatable import Simulatable

from .hvac_properties import HvacNoiseProperties, HvacProperties


class HVACObsDict(HvacProperties):
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
    max_consumption: float
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
        self.init_props.cooling_capacity = random.choice(
            self.noise_props.cooling_capacity_list
        )

    def _get_obs(self) -> HVACObsDict:
        obs_dict = self.init_props.dict()
        obs_dict.update(
            {
                "turned_on": self.turned_on,
                "seconds_since_off": self.seconds_since_off,
                "lockout": self.lockout,
            }
        )
        return HVACObsDict(**obs_dict)

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

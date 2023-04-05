from copy import deepcopy
import random
from datetime import timedelta
from app.core.environment.simulatable import Simulatable
from app.core.environment.environment_properties import (
    EnvironmentObsDict,
    HvacProperties,
)


class HVAC(Simulatable):
    init_props: HvacProperties
    seconds_since_off: int
    lockout: bool
    turned_on: bool

    def __init__(self, hvac_props: HvacProperties) -> None:
        """Initialize HVAC class."""
        self.init_props = deepcopy(hvac_props)
        self.reset()

    def reset(self) -> EnvironmentObsDict:
        """Reset function of the HVAC class."""
        self.seconds_since_off: int = 0
        self.lockout = False
        self.turned_on = True
        return self.get_obs()

    def step(self, action: bool, time_step: timedelta) -> EnvironmentObsDict:
        """Take a step in time for the HVAC, given action of the TCL agent."""
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
        return self.get_obs()

    def apply_noise(self) -> None:
        """Apply noise to hvac initial properties."""
        self.init_props.cooling_capacity = random.choices(
            self.init_props.noise_prop.cooling_capacity_list
        )[0]

    def get_obs(self) -> EnvironmentObsDict:
        """Generate hvac observation dictionnary."""
        obs_dict: EnvironmentObsDict = {
            "turned_on": self.turned_on,
            "seconds_since_off": self.seconds_since_off,
            "lockout": self.lockout,
            "cop": self.init_props.cop,
            "cooling_capacity": self.init_props.cooling_capacity,
            "latent_cooling_fraction": self.init_props.latent_cooling_fraction,
            "lockout_duration": self.init_props.lockout_duration,
        }
        return obs_dict

    def get_heat_transfer(self) -> float:
        """Compute the rate of heat transfer produced by the HVAC.

        Return: q_hvac (float, heat of transfer produced by the HVAC, in Watts).
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

    def get_power_consumption(self) -> float:
        """Compute the electric power consumption of the HVAC.

        Return: power_cons (float, electric power consumption of the HVAC, in Watts).
        """
        if self.turned_on:
            power_cons = self.init_props.max_consumption
        else:
            power_cons = 0.0

        return power_cons

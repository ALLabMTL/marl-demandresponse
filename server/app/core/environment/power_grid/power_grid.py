import random
from datetime import datetime, timedelta
from typing import TypedDict

import numpy as np

from app.core.environment.cluster.cluster import Cluster
from app.core.environment.environment_properties import EnvironmentObsDict
from app.core.environment.power_grid.interpolation import PowerInterpolator
from app.core.environment.power_grid.power_grid_properties import PowerGridProperties
from app.core.environment.power_grid.signal_calculator import SignalCalculator
from app.core.environment.simulatable import Simulatable


class PowerGridObsDict(TypedDict):
    reg_signal: float


class PowerGrid(Simulatable):
    """
    Simulatable object representing a power grid, with functionality to update the power supply based on the current environment and compute a signal. 

    Attributes:
        init_props (PowerGridProperties): The initial properties of the power grid.
        base_power (float): The base power of the power grid.
        current_signal (float): The current signal of the power grid.
        cluster (Cluster): The cluster in which the power grid is situated.
        signal_calculator (SignalCalculator): An object used to compute the current signal of the power grid.
        power_interpolator (PowerInterpolator): An object used to interpolate the power of the power grid.
    """
    init_props: PowerGridProperties
    base_power: float
    current_signal: float
    cluster: Cluster
    signal_calculator: SignalCalculator
    power_interpolator: PowerInterpolator

    def __init__(self, power_grid_props: PowerGridProperties, cluster: Cluster) -> None:
        """Initialize a new instance of the PowerGrid class."""
        #  TODO: use parser service
        self.init_props = power_grid_props
        # Base ratio, randomly multiplying by a number between 1/artificial_signal_ratio_range and artificial_signal_ratio_range, scaled on a logarithmic scale.
        self.init_props.artificial_ratio = (
            self.init_props.artificial_ratio
            * self.init_props.artificial_signal_ratio_range ** (random.random() * 2 - 1)
        )
        self.cluster = cluster
        self.current_signal = (
            self.init_props.base_power_props.avg_power_per_hvac
            * self.cluster.init_props.nb_agents
        )
        self.current_signal = 0.0
        self.signal_calculator = SignalCalculator(
            self.init_props.signal_properties, self.cluster.init_props.nb_agents
        )

        if self.init_props.base_power_props.mode == "interpolation":
            self.power_interpolator = PowerInterpolator(
                self.init_props.base_power_props, self.cluster.init_props.house_prop
            )
            self.time_since_last_interp = (
                self.init_props.base_power_props.interp_update_period + 1
            )

    def reset(self) -> dict:
        """
        Reset the state of the power grid environment.

        Parameters:
            None

        Returns:
            dict: Empty dictionary.
        """
        return {}

    def step(
        self, date_time: datetime, time_step: timedelta, current_od_temp: float
    ) -> EnvironmentObsDict:
        """
        Simulate one step in the power grid environment.

        Parameters:
            date_time (datetime): The current datetime.
            time_step (timedelta): The time delta between the current datetime and the previous one.
            current_od_temp (float): The current outdoor temperature.

        Returns:
            EnvironmentObsDict: A dictionary containing the current regulatory signal.
        """
        self.power_step(date_time, time_step, current_od_temp)
        self.current_signal = self.signal_calculator.compute_signal(
            self.base_power, date_time
        )
        # Artificial_ratio should be 1. Only change for experimental purposes.
        self.current_signal = self.current_signal * self.init_props.artificial_ratio
        self.current_signal = np.minimum(self.current_signal, self.cluster.max_power)

        return self.get_obs()

    def get_obs(self) -> EnvironmentObsDict:
        """
        Get the current observation of the power grid environment.

        Parameters:
            None

        Returns:
            EnvironmentObsDict: A dictionary containing the current regulatory signal.
        """
        obs_dict: EnvironmentObsDict = EnvironmentObsDict(
            reg_signal=self.current_signal
        )
        return obs_dict

    def apply_noise(self) -> None:
        """
        Apply noise to the current regulatory signal.

        Parameters:
            None

        Returns:
            None
        """
        pass

    def power_step(
        self, date_time: datetime, time_step: timedelta, current_od_temp: float
    ) -> None:
        """
        Simulate one step in the power grid environment's power consumption.

        Parameters:
            date_time (datetime): The current datetime.
            time_step (timedelta): The time delta between the current datetime and the previous one.
            current_od_temp (float): The current outdoor temperature.

        Returns:
            None
        """
        if self.init_props.base_power_props.mode == "constant":
            self.base_power = (
                self.init_props.base_power_props.avg_power_per_hvac
                * self.cluster.init_props.nb_agents
            )
        elif self.init_props.base_power_props.mode == "interpolation":
            self.time_since_last_interp += time_step.seconds
            if (
                self.time_since_last_interp
                >= self.init_props.base_power_props.interp_update_period
            ):
                self.base_power = self.power_interpolator.interpolate_power(
                    date_time,
                    current_od_temp,
                    self.init_props.base_power_props.interp_nb_agents,
                    self.cluster.buildings,
                )
                self.time_since_last_interp = 0

import random
from datetime import datetime, timedelta
from typing import TypedDict
import numpy as np

from app.core.environment.power_grid.interpolation import PowerInterpolator
from app.core.environment.power_grid.power_grid_properties import PowerGridProperties
from app.core.environment.simulatable import Simulatable
from app.core.environment.power_grid.signal_calculator import SignalCalculator
from app.core.environment.cluster.cluster import Cluster
from app.core.environment.environment_properties import EnvironmentObsDict


class PowerGridObsDict(TypedDict):
    reg_signal: float


class PowerGrid(Simulatable):
    init_props: PowerGridProperties
    base_power: float
    current_signal: float
    cluster: Cluster
    signal_calculator: SignalCalculator
    power_interpolator: PowerInterpolator

    def __init__(self, power_grid_props: PowerGridProperties, cluster: Cluster) -> None:
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
        return {}

    def step(
        self, date_time: datetime, time_step: timedelta, current_od_temp: float
    ) -> EnvironmentObsDict:
        self.power_step(date_time, time_step, current_od_temp)
        self.current_signal = self.signal_calculator.compute_signal(
            self.base_power, date_time
        )
        # Artificial_ratio should be 1. Only change for experimental purposes.
        self.current_signal = self.current_signal * self.init_props.artificial_ratio
        self.current_signal = np.minimum(self.current_signal, self.cluster.max_power)

        return self.get_obs()

    def get_obs(self) -> EnvironmentObsDict:
        obs_dict: EnvironmentObsDict = EnvironmentObsDict(
            reg_signal=self.current_signal
        )
        return obs_dict

    def apply_noise(self) -> None:
        pass

    def power_step(
        self, date_time: datetime, time_step: timedelta, current_od_temp: float
    ) -> None:
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

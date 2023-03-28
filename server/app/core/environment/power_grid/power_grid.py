import random
from datetime import datetime, timedelta
import numpy as np

from app.core.environment.power_grid.interpolation import PowerInterpolator
from app.core.environment.power_grid.perlin import Perlin
from app.core.environment.power_grid.power_grid_properties import PowerGridProperties
from app.core.environment.simulatable import Simulatable
from app.core.environment.power_grid.signal_calculator import SignalCalculator
from app.core.environment.cluster.cluster import Cluster
from app.core.environment.cluster.building_properties import BuildingProperties


class PowerGrid(Simulatable):
    initial_properties: PowerGridProperties
    power_interpolator: PowerInterpolator
    perlin: Perlin
    cumulated_abs_noise: int
    base_power: float
    nb_steps: int
    max_power: int
    nb_hvacs: int
    time_since_last_interp: int
    current_signal: float
    solar_gain: bool

    def __init__(
        self, max_power: int, nb_hvacs: int, solar_gain: bool, cluster: Cluster
    ) -> None:
        super().__init__()
        #  TODO: use parser service
        self.init_props = PowerGridProperties()
        self.init_building_props = BuildingProperties()
        # Base ratio, randomly multiplying by a number between 1/artificial_signal_ratio_range and artificial_signal_ratio_range, scaled on a logarithmic scale.
        self.init_props.artificial_ratio = (
            self.init_props.artificial_ratio
            * self.init_props.artificial_signal_ratio_range ** (random.random() * 2 - 1)
        )
        self.cluster = cluster
        self.cumulated_abs_noise = 0
        self.nb_steps = 0
        self.max_power = max_power
        self.base_power = 0
        self.nb_hvacs = nb_hvacs
        self.solar_gain = solar_gain
        self.signal_calculator = SignalCalculator(
            self.init_props.signal_properties, self.nb_hvacs
        )

        if self.init_props.base_power_props.mode == "interpolation":
            self.power_interpolator = PowerInterpolator(
                self.init_props.base_power_props.path_datafile,
                self.init_props.base_power_props.path_parameter_dict,
                self.init_props.base_power_props.path_dict_keys,
                self.init_building_props,
            )
            self.time_since_last_interp = (
                self.init_props.base_power_props.interp_update_period + 1
            )

        if self.init_props.signal_properties.mode == "perlin":
            self.perlin = Perlin(
                1,
                self.init_props.signal_properties.nb_octaves,
                self.init_props.signal_properties.octaves_step,
                self.init_props.signal_properties.period,
                random.random(),
            )

    def _reset(self) -> dict:
        return {}

    def _step(
        self, date_time: datetime, time_step: timedelta, current_od_temp: float
    ) -> None:
        self.power_step(date_time, time_step, current_od_temp)
        self.current_signal = self.signal_calculator.compute_signal(
            self.base_power, date_time
        )
        # Artificial_ratio should be 1. Only change for experimental purposes.
        self.current_signal = self.current_signal * self.init_props.artificial_ratio
        self.current_signal = np.minimum(self.current_signal, self.max_power)

        return self.current_signal

    def _get_obs(self) -> dict:
        return {}

    def apply_noise(self) -> None:
        pass

    def power_step(
        self, date_time: datetime, time_step: timedelta, current_od_temp: float
    ) -> None:
        if self.init_props.base_power_props.mode == "constant":
            self.base_power = (
                self.init_props.base_power_props.avg_power_per_hvac * self.nb_hvacs
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
                    self.solar_gain,
                    self.init_props.base_power_props.interp_nb_agents,
                    self.cluster.buildings,
                )
                self.time_since_last_interp = 0

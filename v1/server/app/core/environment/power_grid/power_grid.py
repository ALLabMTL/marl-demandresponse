import csv
import json
import random
import time
from datetime import datetime, timedelta

import numpy as np

from app.core.environment.power_grid.interpolation import PowerInterpolator
from app.core.environment.power_grid.perlin import Perlin
from app.core.environment.power_grid.power_grid_properties import PowerGridProperties
from app.core.environment.simulatable import Simulatable


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

    def __init__(self, max_power: int, nb_hvacs: int, solar_gain: bool) -> None:
        super().__init__()
        self._reset(max_power, nb_hvacs, solar_gain)

    def _reset(self, max_power: int, nb_hvacs: int, solar_gain: bool) -> dict:
        #  TODO: use parser service
        self.initial_properties = PowerGridProperties()
        # Base ratio, randomly multiplying by a number between 1/artificial_signal_ratio_range and artificial_signal_ratio_range, scaled on a logarithmic scale.
        self.initial_properties.artificial_ratio = (
            self.initial_properties.artificial_ratio
            * self.initial_properties.artificial_signal_ratio_range
            ** (random.random() * 2 - 1)
        )
        self.cumulated_abs_noise = 0
        self.nb_steps = 0
        self.max_power = max_power
        self.base_power = 0
        self.nb_hvacs = nb_hvacs
        self.solar_gain = solar_gain

        # if self.initial_properties.base_power_props.mode == "interpolation":
        #     self.init_power_interpolator()

        if "perlin" in self.initial_properties.signal_properties.mode:
            self.perlin = Perlin(
                1,
                self.initial_properties.signal_properties.nb_octaves,
                self.initial_properties.signal_properties.octaves_step,
                self.initial_properties.signal_properties.period,
            )

    def _step(self, date_time: datetime, time_step: timedelta) -> None:
        self.power_step(date_time, time_step)
        self.signal_step(date_time)

        # Artificial_ratio should be 1. Only change for experimental purposes.
        self.current_signal = (
            self.current_signal * self.initial_properties.artificial_ratio
        )
        self.current_signal = np.minimum(self.current_signal, self.max_power)

        return self.current_signal

    def _get_obs(self) -> dict:
        return super()._get_obs()

    def apply_noise(self) -> None:
        return super().apply_noise()

    # TODO: move all this
    def power_step(self, date_time: datetime, time_step: timedelta) -> None:
        mode = {
            "constant": self.constant_base_power(),
            # "interpolation" : self.interpolated_base_power(date_time, time_step)
        }
        mode.get(self.initial_properties.base_power_props.mode)

    def constant_base_power(self) -> None:
        self.base_power = (
            self.initial_properties.base_power_props.avg_power_per_hvac * self.nb_hvacs
        )

    def signal_step(self, date_time: datetime) -> None:
        if "perlin" in (self.initial_properties.base_power_props.mode):
            self.initial_properties.base_power_props.mode = "perlin"
        # TODO: implement other signal shapes
        mode = {
            "flat": self.flat_signal(),
            # "sinusoidals": self.sinusoidals_signal(date_time),
            # "regular_steps": self.regular_steps_signal(date_time),
            # "perlin": self.perlin_signal(date_time)
        }
        mode.get(self.initial_properties.signal_properties.mode)

    def flat_signal(self) -> None:
        self.current_signal = self.base_power

    # def sinusoidals_signal(self, date_time: datetime) -> None:
    #     """Compute the outdoors temperature based on the time, being the sum of several sinusoidal signals"""
    #     amplitudes = [
    #         self.base_power * ratio
    #         for ratio in self.initial_properties.signal_properties.amplitude_ratios
    #     ]
    #     periods = self.initial_properties.signal_properties.periods
    #     if len(periods) != len(amplitudes):
    #         raise ValueError(
    #             "Power grid signal parameters: periods and amplitude_ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(amplitude_ratios): {}.".format(
    #                 len(periods), len(amplitudes)
    #             )
    #         )

    #     time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

    #     signal = self.base_power
    #     for i in range(len(periods)):
    #         signal += amplitudes[i] * np.sin(2 * np.pi * time_sec / periods[i])
    #     self.current_signal = signal

    # def regular_steps_signal(self, date_time: datetime) -> None:
    #     """Compute the outdoors temperature based on the time using pulse width modulation"""
    #     amplitude = self.signal_params["amplitude_per_hvac"] * self.nb_hvacs
    #     ratio = self.base_power / amplitude

    #     period = self.signal_params["period"]

    #     signal = 0
    #     time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

    #     signal = amplitude * np.heaviside(
    #         (time_sec % period) - (1 - ratio) * period, 1
    #     )
    #     self.current_signal = signal

    # def perlin_signal(self, date_time: datetime) -> None:
    #     amplitude = self.initial_properties.signal_properties.amplitude_ratios
    #     # TODO: remove magic number
    #     unix_time_stamp = time.mktime(date_time.timetuple()) % 86400
    #     signal = self.base_power
    #     perlin = self.perlin.calculate_noise(unix_time_stamp)

    #     self.cumulated_abs_noise += np.abs(signal * amplitude * perlin)
    #     self.nb_steps += 1
    #     self.current_signal = np.maximum(0, signal + (signal * amplitude * perlin))

    # def interpolated_base_power(self, date_time: datetime, time_step: timedelta) -> None:
    #     self.time_since_last_interp += time_step.seconds
    #     if self.time_since_last_interp >= self.initial_properties.base_power_props.interp_update_period:
    #         self.base_power = self.interpolate_power(date_time)
    #         self.time_since_last_interp = 0

    # def interpolate_power(self, date_time: datetime) -> None :
    #     # TODO: implement power interpolation
    #     pass

    # def init_power_interpolator(self) -> None:
    #
    #     with open(
    #         self.initial_properties.base_power_props.path_parameter_dict
    #     ) as json_file:
    #         self.interp_parameters_dict = json.load(json_file)
    #     with open(
    #         self.initial_properties.base_power_props.path_dict_keys
    #     ) as f:
    #         reader = csv.reader(f)
    #         self.interp_dict_keys = list(reader)[0]

    #     self.power_interpolator = PowerInterpolator(
    #         self.initial_properties.base_power_props.path_datafile,
    #         self.interp_parameters_dict,
    #         self.interp_dict_keys
    #     )

    #     self.time_since_last_interp = self.initial_properties.base_power_props.interp_update_period + 1

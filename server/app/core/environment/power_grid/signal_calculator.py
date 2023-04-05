from datetime import datetime
import random
import time

import numpy as np
from app.core.environment.power_grid.power_grid_properties import SignalProperties
from app.core.environment.power_grid.perlin import Perlin


class SignalCalculator:
    def __init__(self, signal_props: SignalProperties, nb_agents: int) -> None:
        self.signal_props = signal_props
        self.nb_agents = nb_agents
        if signal_props.mode == "perlin":
            self.perlin = Perlin(
                1,
                self.signal_props.nb_octaves,
                self.signal_props.octaves_step,
                self.signal_props.period,
                random.random(),
            )

    def flat_signal(self, date_time: datetime, base_power: float) -> float:
        return base_power

    def sinusoidals_signal(self, date_time: datetime, base_power: float) -> float:
        """Compute the outdoors temperature based on the time, being the sum of several sinusoidal signals"""
        amplitudes = [
            base_power * ratio for ratio in self.signal_props.amplitude_ratios
        ]

        if len(self.signal_props.periods) != len(amplitudes):
            raise ValueError(
                "Power grid signal parameters: periods and amplitude_ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(amplitude_ratios): {}.".format(
                    len(self.signal_props.periods), len(amplitudes)
                )
            )

        time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

        signal = base_power
        for period, _ in enumerate(self.signal_props.periods):
            signal += amplitudes[period] * np.sin(
                2 * np.pi * time_sec / self.signal_props.periods[period]
            )

        return signal

    def regular_steps_signal(self, date_time: datetime, base_power: float) -> float:
        """Compute the outdoors temperature based on the time using pulse width modulation"""
        amplitude = self.signal_props.amplitude_per_hvac * self.nb_agents
        ratio = base_power / amplitude

        period = self.signal_props.period

        signal = 0
        time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

        signal = amplitude * np.heaviside((time_sec % period) - (1 - ratio) * period, 1)
        return signal

    def perlin_signal(self, date_time: datetime, base_power: float) -> float:
        amplitude = self.signal_props.amplitude_ratios[0]
        unix_time_stamp = time.mktime(date_time.timetuple()) % 86400
        perlin = self.perlin.calculate_noise(unix_time_stamp)
        return np.maximum(0, base_power + (base_power * amplitude * perlin))

    def compute_signal(self, base_power: float, date_time: datetime) -> float:
        mode = getattr(self, (self.signal_props.mode + "_signal"))
        return mode(date_time, base_power)

import random
import time
from datetime import datetime

import numpy as np

from app.core.environment.power_grid.perlin import Perlin
from app.core.environment.power_grid.power_grid_properties import SignalProperties


class SignalCalculator:
    """
    Class that calculates the power grid signal based on the given configuration.

    Parameters:
        signal_props (SignalProperties): The configuration of the power grid signal.
        nb_agents (int): The number of agents that will consume the power grid signal.
    """

    def __init__(self, signal_props: SignalProperties, nb_agents: int) -> None:
        """Initialize a SignalCalculator object."""
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
        """
        Compute the power signal as a flat value. Returns the base power value provided as input.

        Parameters:
            date_time (datetime): The datetime for which to compute the power signal.
            base_power (float): The base power value for the signal.

        Returns:
            signal (float): The power signal value computed as a flat value equal to the base power.
        """
        return base_power

    def sinusoidals_signal(self, date_time: datetime, base_power: float) -> float:
        """
        Compute the outdoors temperature based on the time, being the sum of several sinusoidal signals.

        Parameters:
            date_time (datetime): The datetime for which to compute the power signal.
            base_power (float): The base power value for the signal.

        Returns:
            signal (float): The power signal value computed as the sum of several sinusoidal signals.
        """
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
        """
        Compute the outdoors temperature based on the time using pulse width modulation.

        Parameters:
            date_time (datetime): The datetime for which to compute the power signal.
            base_power (float): The base power value for the signal.

        Returns:
            signal (float): The power signal value computed using pulse width modulation.
        """
        amplitude = self.signal_props.amplitude_per_hvac * self.nb_agents
        ratio = base_power / amplitude

        period = self.signal_props.period

        signal = 0
        time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

        signal = amplitude * np.heaviside((time_sec % period) - (1 - ratio) * period, 1)
        return signal

    def perlin_signal(self, date_time: datetime, base_power: float) -> float:
        """
        Compute the power signal using Perlin noise. The amplitude, period, and number of octaves for the Perlin noise are defined in the SignalProperties object provided to the SignalCalculator constructor.

        Parameters:
            date_time (datetime): The datetime for which to compute the power signal.
            base_power (float): The base power value for the signal.

        Returns:
            signal (float): The power signal value computed using Perlin noise.
        """

        amplitude = self.signal_props.amplitude_ratios[0]
        unix_time_stamp = time.mktime(date_time.timetuple()) % 86400
        perlin = self.perlin.calculate_noise(unix_time_stamp)
        return np.maximum(0, base_power + (base_power * amplitude * perlin))

    def compute_signal(self, base_power: float, date_time: datetime) -> float:
        """
        Compute the power signal for the given base_power and date_time using the mode specified in the SignalProperties object provided to the SignalCalculator constructor. This method delegates the actual computation to one of the four signal methods (flat_signal, sinusoidals_signal, regular_steps_signal, perlin_signal) based on the mode attribute in the SignalProperties object.

        Parameters:
            base_power (float): The base power value for the signal.
            date_time (datetime): The datetime for which to compute the power signal.

        Returns:
            signal (float): The power signal value computed based on the mode attribute in the SignalProperties object.
        """
        mode = getattr(self, (self.signal_props.mode + "_signal"))
        return mode(date_time, base_power)

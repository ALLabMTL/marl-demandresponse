import random
from abc import ABC
from copy import deepcopy
from datetime import datetime, timedelta
from typing import List

import numpy as np

from app.core.environment.cluster.building_properties import (
    BuildingNoiseProperties,
    BuildingProperties,
)
from app.core.environment.cluster.hvac import HVAC
from app.core.environment.simulatable import Simulatable


class Building(Simulatable):
    initial_properties: BuildingProperties
    noise_properties: BuildingNoiseProperties
    indoor_temp: int
    current_mass_temp: int
    current_solar_gain: int
    hvacs: List[HVAC]
    max_consumption: int

    def __init__(self) -> None:
        super().__init__()
        self._reset()

    def _reset(self) -> dict:
        super()._reset()
        # TODO: Initialize values with parser service
        self.initial_properties = BuildingProperties()
        self.noise_properties = BuildingNoiseProperties()
        self.hvacs = list([HVAC()] * self.initial_properties.nb_hvacs)
        self.max_consumption = 0
        for hvac in self.hvacs:
            self.max_consumption += hvac.max_consumption
        self.current_solar_gain = 0
        self.indoor_temp = self.initial_properties.init_air_temp
        self.current_mass_temp = self.initial_properties.init_mass_temp
        return self._get_obs()

    def _step(
        self, od_temp: float, time_step: timedelta, date_time: datetime, action: bool
    ) -> None:
        """
        Take a time step for the house

        Return: -

        Parameters:
        self
        od_temp: float, current outdoors temperature in Celsius
        time_step: timedelta, time step duration
        date_time: datetime, current date and time
        """
        super()._step()
        for hvac in self.hvacs:
            hvac._step(action, time_step)
        self.update_temperature(od_temp, time_step, date_time)

    def _get_obs(self) -> dict:
        super()._get_obs()
        # TODO: this doesnt work with multiple hvacs
        state_dict = {}
        for hvac in self.hvacs:
            state_dict.update(hvac._get_obs())
            state_dict.update(
                self.initial_properties.dict(
                    include={"target_temp", "deadband", "Ua", "Cm", "Ca", "Hm"}
                )
            )
            state_dict.update(
                {
                    "indoor_temp": self.indoor_temp,
                    "mass_temp": self.current_mass_temp,
                    "solar_gain": self.current_solar_gain,
                }
            )
        return state_dict

    def message(self, thermal: bool, hvac: bool) -> dict:
        """
        Message sent by the house to other agents
        """
        # TODO: make this apply to all hvacs
        message = {
            "current_temp_diff_to_target": self.indoor_temp
            - self.initial_properties.target_temp,
            "hvac_seconds_since_off": self.hvacs[0].seconds_since_off,
            "hvac_curr_consumption": self.hvacs[0].get_power_consumption(),
            "hvac_max_consumption": self.hvacs[0].max_consumption,
            "hvac_lockout_duration": self.hvacs[0].init_props.lockout_duration,
        }

        if thermal:
            message.update(
                self.initial_properties.dict(include={"Ua", "Ca", "Hm", "Cm"})
            )
        if hvac:
            message.update(
                self.hvacs[0].initial_properties.dict(exclude={"lockout_duration"})
            )
        return message

    def update_temperature(
        self, od_temp: int, time_step: timedelta, date_time: datetime
    ) -> None:
        """
        Update the temperature of the house

        Return: -

        Parameters:
        self
        od_temp: float, current outdoors temperature in Celsius
        time_step: timedelta, time step duration
        date_time: datetime, current date and time


        ---
        Model taken from http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide
        """
        # TODO: Make it prettier
        time_step_sec = time_step.seconds
        Hm, Ca, Ua, Cm = (
            self.initial_properties.Hm,
            self.initial_properties.Ca,
            self.initial_properties.Ua,
            self.initial_properties.Cm,
        )

        # Convert Celsius temperatures in Kelvin
        od_temp_K = od_temp + 273
        current_temp_K = self.indoor_temp + 273
        current_mass_temp_K = self.current_mass_temp + 273

        # Heat from hvacs (negative if it is AC)
        total_Qhvac = self.hvacs[0].get_Q()

        # Total heat addition to air
        if self.initial_properties.solar_gain:
            self.current_solar_gain = self.compute_solar_gain(date_time)
        else:
            self.current_solar_gain = 0

        other_Qa = self.current_solar_gain  # windows, ...
        Qa = total_Qhvac + other_Qa
        # Heat from inside devices (oven, windows, etc)
        Qm = 0

        # Variables and time constants
        a = Cm * Ca / Hm
        b = Cm * (Ua + Hm) / Hm + Ca
        c = Ua
        d = Qm + Qa + Ua * od_temp_K
        g = Qm / Hm

        r1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        r2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        dTA0dt = (
            Hm * current_mass_temp_K / Ca
            - (Ua + Hm) * current_temp_K / Ca
            + Ua * od_temp_K / Ca
            + Qa / Ca
        )

        A1 = (r2 * current_temp_K - dTA0dt - r2 * d / c) / (r2 - r1)
        A2 = current_temp_K - d / c - A1
        A3 = r1 * Ca / Hm + (Ua + Hm) / Hm
        A4 = r2 * Ca / Hm + (Ua + Hm) / Hm

        # Updating the temperature
        new_current_temp_K = (
            A1 * np.exp(r1 * time_step_sec) + A2 * np.exp(r2 * time_step_sec) + d / c
        )
        new_current_mass_temp_K = (
            A1 * A3 * np.exp(r1 * time_step_sec)
            + A2 * A4 * np.exp(r2 * time_step_sec)
            + g
            + d / c
        )

        self.indoor_temp = new_current_temp_K - 273
        self.current_mass_temp = new_current_mass_temp_K - 273

    def apply_noise(self) -> None:
        # TODO: make this prettier
        # Gaussian noise: target temp
        self.initial_properties.init_air_temp += abs(
            random.gauss(0, self.noise_properties.std_start_temp)
        )

        self.initial_properties.init_mass_temp += abs(
            random.gauss(0, self.noise_properties.std_start_temp)
        )
        self.initial_properties.target_temp += abs(
            random.gauss(0, self.noise_properties.std_target_temp)
        )

        # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance
        factor_Ua = random.triangular(
            self.noise_properties.factor_thermo_low,
            self.noise_properties.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.initial_properties.Ua = factor_Ua

        factor_Cm = random.triangular(
            self.noise_properties.factor_thermo_low,
            self.noise_properties.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.initial_properties.Cm *= factor_Cm

        factor_Ca = random.triangular(
            self.noise_properties.factor_thermo_low,
            self.noise_properties.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.initial_properties.Ca *= factor_Ca

        factor_Hm = random.triangular(
            self.noise_properties.factor_thermo_low,
            self.noise_properties.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.initial_properties.Hm *= factor_Hm

        for hvac in self.hvacs:
            hvac.apply_noise()

    def compute_solar_gain(self, date_time: datetime) -> float:
        """
        Computes the solar gain, i.e. the heat transfer received from the sun through the windows.

        Return:
        solar_gain: float, direct solar radiation passing through the windows at a given moment in Watts

        Parameters
        date_time: datetime, current date and time

        ---
        Source and assumptions:
        CIBSE. (2015). Environmental Design - CIBSE Guide A (8th Edition) - 5.9.7 Solar Cooling Load Tables. CIBSE.
        Retrieved from https://app.knovel.com/hotlink/pdf/id:kt0114THK9/environmental-design/solar-cooling-load-tables
        Table available: https://www.cibse.org/Knowledge/Guide-A-2015-Supplementary-Files/Chapter-5

        Coefficient obtained by performing a polynomial regression on the table "solar cooling load at stated sun time at latitude 30".

        Based on the following assumptions.
        - Latitude is 30. (The latitude of Austin in Texas is 30.266666)
        - The SCL before 7:30 and after 17:30 is negligible for latitude 30.
        - The windows are distributed perfectly evenly around the building.
        - There are no horizontal windows, for example on the roof.
        """

        x = date_time.hour + date_time.minute / 60 - 7.5
        if x < 0 or x > 10:
            solar_cooling_load = 0
        else:
            y = date_time.month + date_time.day / 30 - 1
            coeff = [
                4.36579418e01,
                1.58055357e02,
                8.76635241e01,
                -4.55944821e01,
                3.24275366e00,
                -4.56096472e-01,
                -1.47795612e01,
                4.68950855e00,
                -3.73313090e01,
                5.78827663e00,
                1.04354810e00,
                2.12969604e-02,
                2.58881400e-03,
                -5.11397219e-04,
                1.56398008e-02,
                -1.18302764e-01,
                -2.71446436e-01,
                -3.97855577e-02,
            ]

            solar_cooling_load = (
                coeff[0]
                + x * coeff[1]
                + y * coeff[2]
                + x**2 * coeff[3]
                + x**2 * y * coeff[4]
                + x**2 * y**2 * coeff[5]
                + y**2 * coeff[6]
                + x * y**2 * coeff[7]
                + x * y * coeff[8]
                + x**3 * coeff[9]
                + y**3 * coeff[10]
                + x**3 * y * coeff[11]
                + x**3 * y**2 * coeff[12]
                + x**3 * y**3 * coeff[13]
                + x**2 * y**3 * coeff[14]
                + x * y**3 * coeff[15]
                + x**4 * coeff[16]
                + y**4 * coeff[17]
            )

        solar_gain = (
            self.initial_properties.window_area
            * self.initial_properties.shading_coeff
            * solar_cooling_load
        )
        return solar_gain

    def get_power_consumption(self) -> int:
        power_consumption = 0
        for hvac in self.hvacs:
            power_consumption += hvac.get_power_consumption()
        return power_consumption

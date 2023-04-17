import random
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np

from app.core.environment.cluster.hvac import HVAC
from app.core.environment.environment_properties import (
    BuildingMessage,
    BuildingProperties,
    EnvironmentObsDict,
)
from app.core.environment.simulatable import Simulatable
from app.utils.utils import compute_solar_gain


class Building(Simulatable):
    """
    Simulatable object representing a building with HVAC system.

    Attributes:
        init_props (BuildingProperties): Initial properties of the building.
        current_mass_temp (float): Current temperature of the building mass in Celsius.
        indoor_temp (float): Current indoor air temperature in Celsius.
        hvac (HVAC): HVAC object.
        max_consumption (float): Maximum power consumption of the HVAC system in Watts.
        current_solar_gain (float): Current solar gain of the building in Watts.
    """


    init_props: BuildingProperties
    current_mass_temp: float
    indoor_temp: float
    hvac: HVAC
    max_consumption: float
    current_solar_gain: float

    def __init__(self, building_props: BuildingProperties) -> None:
        """
        Constructor for the Building class.

        Parameters:
            building_props: A BuildingProperties object that contains the properties of the building.
        Returns:
            None
        """
        self.init_props = deepcopy(building_props)
        self.reset()

    def reset(self) -> EnvironmentObsDict:
        """
        Resets the state of the building.

        Returns:
            An EnvironmentObsDict object that contains the state of the building.
        """
        self.hvac = HVAC(self.init_props.hvac_prop)
        self.max_consumption = self.hvac.init_props.max_consumption
        self.current_solar_gain = 0.0
        self.current_mass_temp = self.init_props.init_mass_temp
        self.indoor_temp = self.init_props.init_air_temp
        return self.get_obs()

    def step(
        self, od_temp: float, time_step: timedelta, date_time: datetime, action: bool
    ) -> None:
        """Take a time step for the building.

        Return: -

        Parameters:
        self
        od_temp: float, current outdoors temperature in Celsius
        time_step: timedelta, time step duration
        date_time: datetime, current date and time
        """
        self.hvac.step(action, time_step)
        self.update_temperature(od_temp, time_step, date_time)

    def get_obs(self) -> EnvironmentObsDict:
        """
        Generate building observation dictionnary.

        Returns:
            An EnvironmentObsDict object that contains the current observation of the building.
        """
        state_dict: EnvironmentObsDict = self.hvac.get_obs()
        state_dict.update(
            {
                "target_temp": self.init_props.target_temp,
                "deadband": self.init_props.deadband,
                "Ua": self.init_props.Ua,
                "Ca": self.init_props.Ca,
                "Cm": self.init_props.Cm,
                "Hm": self.init_props.Hm,
                "indoor_temp": self.indoor_temp,
                "mass_temp": self.current_mass_temp,
                "solar_gain": self.current_solar_gain,
            }
        )
        return state_dict

    def message(self, thermal_message: bool, hvac_message: bool) -> BuildingMessage:
        """
        Message sent by the building to other agents.
        
        Parameters:
            thermal_message: A boolean indicating whether to include thermal properties in the message.
            hvac_message: A boolean indicating whether to include HVAC properties in the message.
        
        Returns:
            A BuildingMessage object that contains the message to be sent by the building.
        """
        message: BuildingMessage = {
            "seconds_since_off": self.hvac.seconds_since_off,
            "curr_consumption": self.hvac.get_power_consumption(),
            "max_consumption": self.hvac.init_props.max_consumption,
            "lockout_duration": self.hvac.init_props.lockout_duration,
            "current_temp_diff_to_target": self.indoor_temp
            - self.init_props.target_temp,
        }

        if hvac_message:
            message.update(
                {
                    "cop": self.hvac.init_props.cop,
                    "latent_cooling_fraction": self.hvac.init_props.latent_cooling_fraction,
                    "cooling_capacity": self.hvac.init_props.cooling_capacity,
                }
            )
        if thermal_message:
            message.update(
                {
                    "Ca": self.init_props.Ca,
                    "Ua": self.init_props.Ua,
                    "Cm": self.init_props.Cm,
                    "Hm": self.init_props.Hm,
                }
            )
        return message

    def update_temperature(
        self, od_temp: float, time_step: timedelta, date_time: datetime
    ) -> None:
        """
        Update the temperature of the house.

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
            self.init_props.Hm,
            self.init_props.Ca,
            self.init_props.Ua,
            self.init_props.Cm,
        )

        # Convert Celsius temperatures in Kelvin
        od_temp_K = od_temp + 273
        current_temp_K = self.indoor_temp + 273
        current_mass_temp_K = self.current_mass_temp + 273

        # Heat from hvacs (negative if it is AC)
        total_Qhvac = self.hvac.get_heat_transfer()

        # Total heat addition to air
        if self.init_props.solar_gain:
            self.current_solar_gain = compute_solar_gain(
                date_time, self.init_props.window_area, self.init_props.shading_coeff
            )
        else:
            self.current_solar_gain = 0.0

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
        new_current_temp_k = (
            A1 * np.exp(r1 * time_step_sec) + A2 * np.exp(r2 * time_step_sec) + d / c
        )
        new_current_mass_temp_k = (
            A1 * A3 * np.exp(r1 * time_step_sec)
            + A2 * A4 * np.exp(r2 * time_step_sec)
            + g
            + d / c
        )

        self.indoor_temp = new_current_temp_k - 273
        self.current_mass_temp = new_current_mass_temp_k - 273

    def apply_noise(self) -> None:
        """Applies noise to the initial building properties gave in config."""
        # TODO: make this prettier
        # Gaussian noise: target temp
        self.init_props.init_air_temp += abs(
            random.gauss(0, self.init_props.noise_prop.std_start_temp)
        )

        self.init_props.init_mass_temp += abs(
            random.gauss(0, self.init_props.noise_prop.std_start_temp)
        )
        self.init_props.target_temp += abs(
            random.gauss(0, self.init_props.noise_prop.std_target_temp)
        )

        # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance
        factor_ua = random.triangular(
            self.init_props.noise_prop.factor_thermo_low,
            self.init_props.noise_prop.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.init_props.Ua = factor_ua

        factor_cm = random.triangular(
            self.init_props.noise_prop.factor_thermo_low,
            self.init_props.noise_prop.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.init_props.Cm *= factor_cm

        factor_ha = random.triangular(
            self.init_props.noise_prop.factor_thermo_low,
            self.init_props.noise_prop.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.init_props.Ca *= factor_ha

        factor_hm = random.triangular(
            self.init_props.noise_prop.factor_thermo_low,
            self.init_props.noise_prop.factor_thermo_high,
            1,
        )  # low, high, mode ->  low <= N <= high, with max prob at mode.
        self.init_props.Hm *= factor_hm
        self.hvac.apply_noise()

    def get_power_consumption(self) -> float:
        """Return current power consumption of Building."""
        return self.hvac.get_power_consumption()

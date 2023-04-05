import random
from copy import deepcopy
from datetime import timedelta
from typing import Dict, List

import numpy as np

from app.core.environment.cluster.cluster import Cluster
from app.core.environment.environment_properties import (
    EnvironmentObsDict,
    EnvironmentProperties,
)
from app.core.environment.power_grid.power_grid import PowerGrid
from app.core.environment.rewards_calculator import RewardsCalculator

DAYS_IN_YEAR = 364.0
SECONDS_IN_MINUTE = 60.0
MINUTES_IN_HOUR = 60.0
HOURS_IN_DAY = 24.0
SECONDS_IN_DAY = SECONDS_IN_MINUTE * MINUTES_IN_HOUR * HOURS_IN_DAY


class Environment:
    """

    The Environment class implements the environment for a building cluster and its power grid.
    The environment provides an interface for simulating the operation of a building cluster with a power grid.

    Attributes:
        init_props (EnvironmentProperties): An object containing the initial properties of the environment.
        cluster (Cluster): An object representing the cluster of buildings.
        date_time (datetime): A datetime object representing the current date and time in the environment.
        current_od_temp (float): A float representing the current outdoor temperature in the environment.
        power_grid (PowerGrid): An object representing the power grid.
        rewards_calculator (RewardsCalculator): An object representing the rewards calculator.

    """
    def __init__(self, env_props: EnvironmentProperties) -> None:
        """
        Initialize a new instance of the Environment class with the provided environment properties.

        Parameters:
            env_props: EnvironmentProperties, the environment properties used to initialize the Environment instance.
        """
        self.init_props = deepcopy(env_props)
        self.reset()

    def reset(self) -> Dict[int, EnvironmentObsDict]:
        """
        Reset the Environment instance to its initial state.

        Returns:
            obs_dict: Dict[int, EnvironmentObsDict], a dictionary of observation dictionaries for each building in the cluster.
        """
        self.cluster = Cluster(self.init_props.cluster_prop)
        self.date_time = self.init_props.start_datetime
        self.apply_noise()
        self.compute_od_temp()
        self.power_grid = PowerGrid(
            self.init_props.power_grid_prop,
            self.cluster,
        )
        self.rewards_calculator = RewardsCalculator(
            self.init_props.reward_prop, self.init_props.cluster_prop.house_prop
        )
        self.power_grid.step(
            self.date_time, self.init_props.time_step, self.current_od_temp
        )
        return self.get_obs()

    def step(
        self, action_dict: Dict[int, bool]
    ) -> tuple[Dict[int, EnvironmentObsDict], Dict[int, float]]:
        """
        Advance the simulation one time step.

        Parameters:
            action_dict: Dict[int, bool], a dictionary of binary actions for each building in the cluster.

        Returns:
            - obs_dict: Dict[int, EnvironmentObsDict], a dictionary of observation dictionaries for each building in the cluster.
            - rewards_dict: Dict[int, float], a dictionary of reward values for each building in the cluster.

        """
        # Step in time
        self.date_time += self.init_props.time_step
        # Cluster step
        self.cluster.step(
            self.current_od_temp, action_dict, self.date_time, self.init_props.time_step
        )

        # Compute outdoor temperature before power grid step
        self.compute_od_temp()

        # Compute reward with the old grid signal
        rewards_dict = self.rewards_calculator.compute_rewards(
            self.cluster.buildings,
            self.cluster.current_power_consumption,
            self.power_grid.current_signal,
        )

        # Power grid step
        self.power_grid.step(
            self.date_time, self.init_props.time_step, self.current_od_temp
        )

        return self.get_obs(), rewards_dict

    def get_obs(self) -> Dict[int, EnvironmentObsDict]:
        """
        Return the current observations of the Environment instance.

        Returns:
            obs_dict: Dict[int, EnvironmentObsDict], a dictionary of observation dictionaries for each building in the cluster.

        """
        obs_list: List[EnvironmentObsDict] = self.cluster.get_obs()
        obs_dict: Dict[int, EnvironmentObsDict] = {}
        for building_id, building in enumerate(obs_list):
            building.update(
                {
                    "OD_temp": self.current_od_temp,
                    "datetime": self.date_time,
                }
            )
            building.update(self.power_grid.get_obs())
            obs_dict.update({building_id: building})

        return obs_dict

    def compute_od_temp(self) -> None:
        """
        Compute the outdoors temperature based on the time, according to a sinusoidal model and add a gaussian random factor.

        Parameters:
            self
             
        Returns:
            None
        """

        # Sinusoidal model
        amplitude = (
            self.init_props.temp_prop.day_temp - self.init_props.temp_prop.night_temp
        ) / 2.0
        bias = (
            self.init_props.temp_prop.day_temp + self.init_props.temp_prop.night_temp
        ) / 2.0
        delay = -6.0 + self.init_props.temp_prop.phase  # Temperature is coldest at 6am
        time_day = self.date_time.hour + self.date_time.minute / SECONDS_IN_MINUTE

        temperature = (
            amplitude * np.sin(2 * np.pi * (time_day + delay) / HOURS_IN_DAY) + bias
        )

        # Adding noise
        temperature += random.gauss(0, self.init_props.temp_prop.temp_std)
        self.current_od_temp = temperature

    def apply_noise(self) -> None:
        """
        Apply noise to the environment by randomizing the start date and adding noise to the cluster.
    
        This method applies noise to the environment by calling the `randomize_date()` method to randomize the start date, and then calling the `apply_noise()` method of the `Cluster` object to add noise to the cluster.

        Parameters:
            self
             
        Returns:
            None
        """
        self.randomize_date()
        self.cluster.apply_noise()

    def randomize_date(self) -> None:
        """
        Randomize the start date of the environment based on the `start_datetime` and `start_datetime_mode` properties of the `EnvironmentProperties` object.
    
        If the `start_datetime_mode` property is set to "random", the start date is randomized by selecting a random number of days and seconds within the year and adding them to the `start_datetime` property. If the `start_datetime_mode` property is set to "fixed", the `start_datetime` property is left unchanged.

        Parameters:
            self
             
        Returns:
            None

        """
        if self.init_props.start_datetime_mode == "random":
            random_days = random.randrange(int(DAYS_IN_YEAR))
            random_seconds = random.randrange(int(SECONDS_IN_DAY))
            self.date_time = self.init_props.start_datetime + timedelta(
                days=random_days, seconds=random_seconds
            )

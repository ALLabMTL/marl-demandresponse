from copy import deepcopy
import random
from typing import Dict, List
from datetime import timedelta

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
    def __init__(self, env_props: EnvironmentProperties) -> None:
        self.init_props = deepcopy(env_props)
        self.reset()

    def reset(self) -> Dict[int, EnvironmentObsDict]:
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
        """Compute the outdoors temperature based on the time, according to a sinusoidal model and add a gaussian random factor.

        Returns:
        temperature: float, outdoors temperature, in Celsius.

        Parameters:
        self
        date_time: datetime, current date and time.

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
        self.randomize_date()
        self.cluster.apply_noise()

    def randomize_date(self) -> None:
        if self.init_props.start_datetime_mode == "random":
            random_days = random.randrange(int(DAYS_IN_YEAR))
            random_seconds = random.randrange(int(SECONDS_IN_DAY))
            self.date_time = self.init_props.start_datetime + timedelta(
                days=random_days, seconds=random_seconds
            )

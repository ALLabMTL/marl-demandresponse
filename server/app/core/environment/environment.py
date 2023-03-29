import random
from datetime import datetime, timedelta
from typing import Dict

import numpy as np

from app.core.environment.cluster.cluster import Cluster
from app.core.environment.cluster.cluster_properties import TemperatureProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.core.environment.power_grid.power_grid import PowerGrid
from app.core.environment.cluster.building_properties import BuildingProperties
from app.core.environment.rewards_calculator import RewardsCalculator

DAYS_IN_YEAR = 364
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
HOURS_IN_DAY = 24
SECONDS_IN_DAY = SECONDS_IN_MINUTE * MINUTES_IN_HOUR * HOURS_IN_DAY


class Environment:
    init_props: EnvironmentProperties
    temp_props: TemperatureProperties
    cluster: Cluster
    power_grid: PowerGrid
    date_time: datetime
    current_od_temp: float

    def __init__(self, env_props: EnvironmentProperties) -> None:
        self.init_props = env_props
        self.date_time = self.init_props.start_datetime
        # TODO: compute phase inside TemperatureProperties dataclass
        # TODO: get properties from parser_service, needs to be changed
        self.default_building_props = BuildingProperties()
        self.cluster = Cluster()
        self.power_grid = PowerGrid(
            self.cluster.max_power,
            self.cluster.nb_hvacs,
            self.cluster.buildings[0].initial_properties.solar_gain,
            self.cluster,
        )
        self.rewards_calculator = RewardsCalculator(
            self.init_props.reward_properties, self.default_building_props
        )
        self.compute_od_temp()

    def _reset(self) -> dict:
        self.build_environment(self.init_props)
        return self._get_obs()

    def _step(
        self, action_dict: Dict[int, bool]
    ) -> tuple[Dict[int, dict], Dict[int, float]]:
        # Step in time
        self.date_time += self.init_props.time_step

        # Cluster step
        self.cluster._step(
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
        self.power_grid._step(
            self.date_time, self.init_props.time_step, self.current_od_temp
        )

        return self._get_obs(), rewards_dict

    def _get_obs(self) -> Dict[int, dict]:
        obs_dict = self.cluster._get_obs()
        for i in range(self.cluster.nb_hvacs):
            obs_dict[i].update(
                {
                    "OD_temp": self.current_od_temp,
                    "datetime": self.date_time,
                    "reg_signal": self.power_grid.current_signal,
                }
            )
        return obs_dict

    def build_environment(self, env_props: EnvironmentProperties) -> None:
        self.init_props = env_props
        self.temp_properties = env_props.cluster_prop.temp_parameters
        self.cluster = Cluster()
        self.apply_noise()
        self.date_time = self.init_props.start_datetime
        self.compute_od_temp()
        self.power_grid = PowerGrid(
            self.cluster.max_power,
            self.cluster.nb_hvacs,
            self.cluster.buildings[0].initial_properties.solar_gain,
            self.cluster,
        )
        self.power_grid._step(
            self.date_time, self.init_props.time_step, self.current_od_temp
        )

    def compute_od_temp(self) -> None:
        """
        Compute the outdoors temperature based on the time, according to a model

        Returns:
        temperature: float, outdoors temperature, in Celsius.

        Parameters:
        self
        date_time: datetime, current date and time.

        """
        # Sinusoidal model
        amplitude = (
            self.temp_properties.day_temp - self.temp_properties.night_temp
        ) / 2
        bias = (self.temp_properties.day_temp + self.temp_properties.night_temp) / 2
        delay = -6 + self.temp_properties.phase  # Temperature is coldest at 6am
        time_day = self.date_time.hour + self.date_time.minute / SECONDS_IN_MINUTE

        temperature = (
            amplitude * np.sin(2 * np.pi * (time_day + delay) / HOURS_IN_DAY) + bias
        )

        # Adding noise
        temperature += random.gauss(0, self.temp_properties.temp_std)
        self.current_od_temp = temperature

    def apply_noise(self) -> None:
        self.randomize_date()
        self.cluster.apply_noise()

    def randomize_date(self) -> None:
        if self.init_props.start_datetime_mode == "random":
            random_days = random.randrange(DAYS_IN_YEAR)
            random_seconds = random.randrange(SECONDS_IN_DAY)
            self.date_time = self.init_props.start_datetime + timedelta(
                days=random_days, seconds=random_seconds
            )

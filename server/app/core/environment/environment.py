import random
from datetime import datetime, timedelta
from typing import Dict

import numpy as np

from app.core.environment.cluster.building import Building
from app.core.environment.cluster.cluster import Cluster
from app.core.environment.cluster.cluster_properties import TemperatureProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.core.environment.power_grid.power_grid import PowerGrid
from app.utils.utils import deadbandL2


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
        self.temp_properties = TemperatureProperties()
        self.cluster = Cluster(env_props.cluster_prop)
        self.power_grid = PowerGrid(
            self.cluster.max_power,
            self.cluster.nb_hvacs,
            self.cluster.buildings[0].initial_properties.solar_gain,
        )
        self.compute_od_temp()

    def _reset(self) -> dict:
        self.build_environment()
        return self._get_obs()

    def _step(self, action_dict: Dict[int, dict]):
        self.date_time += self.init_props.time_step
        # Cluster step
        self.cluster._step(
            self.current_od_temp, action_dict, self.date_time, self.init_props.time_step
        )

        self.compute_od_temp()

        # Compute reward with the old grid signal
        rewards_dict = self.compute_rewards()

        # Power grid step
        self.power_grid._step(self.date_time, self.init_props.time_step)

        return self._get_obs(), rewards_dict

    def _get_obs(self) -> Dict[int, dict]:
        """"""
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

    def build_environment(self) -> None:
        self.init_props = EnvironmentProperties()
        self.temp_properties = TemperatureProperties()
        self.cluster = Cluster(self.init_props.cluster_prop)
        self.apply_noise()
        self.date_time = self.init_props.start_datetime
        self.compute_od_temp()
        self.power_grid = PowerGrid(
            self.cluster.max_power,
            self.cluster.nb_hvacs,
            self.cluster.buildings[0].initial_properties.solar_gain,
        )
        self.power_grid._step(self.date_time, self.init_props.start_datetime)
        # TODO: compute OD_temp after step of cluster

    def compute_rewards(self):
        """
        Compute the reward of each TCL agent

        Returns:
        rewards_dict: a dictionary, containing the rewards of each TCL agent.

        Parameters:
        temp_penalty_dict: a dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """

        # TODO: Make it prettier
        rewards_dict: dict[str, float] = {}
        signal_penalty = self.reg_signal_penalty()

        default_building = Building()

        norm_temp_penalty = deadbandL2(
            default_building.initial_properties.target_temp,
            0,
            default_building.initial_properties.target_temp + 1,
        )

        norm_sig_penalty = deadbandL2(
            self.init_props.reward_prop.norm_reg_sig,
            0,
            0.75 * self.init_props.reward_prop.norm_reg_sig,
        )

        # Temperature penalties
        temp_penalty_dict = {}
        for building_id, _ in enumerate(self.cluster.buildings):
            temp_penalty_dict[building_id] = self.compute_temp_penalty(building_id)

            rewards_dict[building_id] = -1 * (
                self.init_props.reward_prop.alpha_temp
                * temp_penalty_dict[building_id]
                / norm_temp_penalty
                + self.init_props.reward_prop.alpha_sig
                * signal_penalty
                / norm_sig_penalty
            )
        return rewards_dict

    def compute_temp_penalty(self, one_house_id: int) -> float:
        """
        Returns: a float, representing the positive penalty due to distance between the target (indoors) temperature and the indoors temperature in a house.

        Parameters:
        target_temp: a float. Target indoors air temperature, in Celsius.
        deadband: a float. Margin of tolerance for indoors air temperature difference, in Celsius.
        house_temp: a float. Current indoors air temperature, in Celsius
        """
        temp_penalty_mode = self.init_props.reward_prop.penalty_props.mode
        temperature_penalty = 0
        if temp_penalty_mode == "individual_L2":
            building = self.cluster.buildings[one_house_id]
            temperature_penalty = deadbandL2(
                building.initial_properties.target_temp,
                building.initial_properties.deadband,
                building.indoor_temp,
            )

            # temperature_penalty = np.clip(temperature_penalty, 0, 20)

        elif temp_penalty_mode == "common_L2":
            ## Mean of all houses L2
            for building in self.cluster.buildings:
                building_temperature_penalty = deadbandL2(
                    building.initial_properties.target_temp,
                    building.initial_properties.deadband,
                    building.indoor_temp,
                )
                temperature_penalty += building_temperature_penalty / len(
                    self.cluster.buildings
                )

        # elif temp_penalty_mode == "common_max":
        #     temperature_penalty = 0
        #     for house_id in self.agent_ids:
        #         house = self.cluster.houses[house_id]
        #         house_temperature_penalty = deadbandL2(
        #             house.target_temp, house.deadband, house.current_temp
        #         )
        #         if house_temperature_penalty > temperature_penalty:
        #             temperature_penalty = house_temperature_penalty

        # elif temp_penalty_mode == "mixture":
        #     temp_penalty_params = self.default_env_prop["reward_prop"][
        #         "temp_penalty_parameters"
        #     ][temp_penalty_mode]

        #     ## Common and max penalties
        #     common_L2 = 0
        #     common_max = 0
        #     for house_id in self.agent_ids:
        #         house = self.cluster.houses[house_id]
        #         house_temperature_penalty = deadbandL2(
        #             house.target_temp, house.deadband, house.current_temp
        #         )
        #         if house_id == one_house_id:
        #             ind_L2 = house_temperature_penalty
        #         common_L2 += house_temperature_penalty / self.nb_agents
        #         if house_temperature_penalty > common_max:
        #             common_max = house_temperature_penalty

        #     ## Putting together
        #     alpha_ind_L2 = temp_penalty_params["alpha_ind_L2"]
        #     alpha_common_L2 = temp_penalty_params["alpha_common_L2"]
        #     alpha_common_max = temp_penalty_params["alpha_common_max"]
        #     temperature_penalty = (
        #         alpha_ind_L2 * ind_L2
        #         + alpha_common_L2 * common_L2
        #         + alpha_common_max * common_max
        #     ) / (alpha_ind_L2 + alpha_common_L2 + alpha_common_max)

        else:
            raise ValueError(
                "Unknown temperature penalty mode: {}".format(temp_penalty_mode)
            )

        return temperature_penalty

    def reg_signal_penalty(self) -> float:
        """
        Returns: a float, representing the positive penalty due to the distance between the regulation signal and the total power used by the TCLs.

        Parameters:
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """
        sig_penalty_mode = self.init_props.reward_prop.penalty_props.mode

        if sig_penalty_mode == "common_L2":
            penalty = (
                (
                    self.cluster.current_power_consumption
                    - self.power_grid.current_signal
                )
                / self.cluster.nb_agents
            ) ** 2
        else:
            raise ValueError(f"Unknown signal penalty mode: {sig_penalty_mode}")

        return penalty

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
        time_day = self.date_time.hour + self.date_time.minute / 60.0

        temperature = amplitude * np.sin(2 * np.pi * (time_day + delay) / 24) + bias

        # Adding noise
        temperature += random.gauss(0, self.temp_properties.temp_std_deviation)
        self.current_od_temp = temperature

    def apply_noise(self) -> None:
        self.randomize_date()
        self.cluster.apply_noise()

    def randomize_date(self):
        if self.init_props.start_datetime_mode == "random":
            DAYS_IN_YEAR = 364
            SECONDS_IN_DAY = 60 * 60 * 24
            random_days = random.randrange(DAYS_IN_YEAR)
            random_seconds = random.randrange(SECONDS_IN_DAY)
            self.date_time = self.init_props.start_datetime + timedelta(
                days=random_days, seconds=random_seconds
            )

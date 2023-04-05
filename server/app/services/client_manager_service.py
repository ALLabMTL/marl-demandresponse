from datetime import datetime
from typing import Dict, List, Tuple, Union
from .socket_manager_service import SocketManager

import numpy as np
import pandas as pd

GRAPH_MEMORY = 5000


class ClientManagerService:
    description: Dict[int, Dict[str, str]]
    temp_diff: np.ndarray
    temp_err: np.ndarray
    air_temp: np.ndarray
    mass_temp: np.ndarray
    target_temp: np.ndarray
    outdoor_temp: np.ndarray
    signal: np.ndarray
    consumption: np.ndarray
    recent_signal: np.ndarray
    recent_consumption: np.ndarray
    data_frame: pd.DataFrame
    houses_data: Dict[int, List[Dict[str, Union[str, float]]]]

    def __init__(self, socket_manager_service: SocketManager) -> None:
        self.socket_manager_service = socket_manager_service
        self.initialize_data()

    def initialize_data(self) -> None:
        self.description = {}
        self.temp_diff = np.array([])
        self.temp_err = np.array([])
        self.air_temp = np.array([])
        self.mass_temp = np.array([])
        self.target_temp = np.array([])
        self.outdoor_temp = np.array([])
        self.signal = np.array([])
        self.consumption = np.array([])
        self.recent_signal = np.array([])
        self.recent_consumption = np.array([])
        self.data_frame = pd.DataFrame()
        self.houses_data = {}

    def update_data(
        self,
        obs_dict: Dict[int, List[Union[float, str, bool, datetime]]],
        time_step: int,
    ) -> Tuple[Dict[str, str], List[Dict[str, Union[str, float]]]]:
        self.data_frame = pd.DataFrame(obs_dict).transpose()
        self.data_frame["temperature_difference"] = (
            self.data_frame["indoor_temp"] - self.data_frame["target_temp"]
        )
        self.data_frame["temperature_error"] = np.abs(
            self.data_frame["indoor_temp"] - self.data_frame["target_temp"]
        )
        self.update_graph_data()
        self.update_desc_data(time_step)
        self.update_houses_data(obs_dict, time_step)
        return self.description[time_step], self.houses_data[time_step]

    def update_graph_data(self) -> None:
        self.temp_diff = np.append(
            self.temp_diff, self.data_frame["temperature_difference"].mean()
        )
        self.temp_err = np.append(
            self.temp_err, self.data_frame["temperature_error"].mean()
        )
        self.air_temp = np.append(self.air_temp, self.data_frame["indoor_temp"].mean())
        self.mass_temp = np.append(self.mass_temp, self.data_frame["mass_temp"].mean())
        self.target_temp = np.append(
            self.target_temp, self.data_frame["target_temp"].mean()
        )
        self.outdoor_temp = np.append(
            self.outdoor_temp, self.data_frame["OD_temp"].mean()
        )
        self.signal = np.append(self.signal, self.data_frame["reg_signal"][0])
        self.consumption = np.append(
            self.consumption, self.data_frame["cluster_hvac_power"][0]
        )
        # We probably wont need this since graphs are made in client side
        self.recent_signal = self.signal[max(-50, -len(self.signal)) :]
        self.recent_consumption = self.consumption[max(-50, -len(self.consumption)) :]

    def update_desc_data(self, time_step: int) -> None:
        # TODO: refactor this
        description = {}
        description["Number of HVAC"] = str(self.data_frame.shape[0])
        description["Number of locked HVAC"] = str(
            np.where(
                self.data_frame["lockout"] & (self.data_frame["turned_on"] == False),
                1,
                0,
            ).sum()
        )
        description["Outdoor temperature"] = (
            str(round(self.data_frame["OD_temp"][0], 2)) 
        )
        description["Mass temperature"] = (
            str(round(self.data_frame["mass_temp"][0], 2)) 
        )
        description["Target temperature"] = (
            str(round(self.data_frame["target_temp"][0], 2)) 
        )
        description["Average indoor temperature"] = (
            str(round(self.data_frame["indoor_temp"].mean(), 2)) 
        )
        description["Average temperature difference"] = (
            str(round(self.data_frame["temperature_difference"].mean(), 2)) 
        )
        description["Regulation signal"] = str(self.data_frame["reg_signal"][0])
        description["Current consumption"] = str(
            self.data_frame["cluster_hvac_power"][0]
        )
        description["Consumption error (%)"] = "{:.3f}%".format(
            (
                self.data_frame["reg_signal"][0]
                - self.data_frame["cluster_hvac_power"][0]
            )
            / self.data_frame["reg_signal"][0]
            * 100
        )
        description["Average temperature error"] = "{:.2f}".format(
                np.mean(self.temp_err[max(-GRAPH_MEMORY, -len(self.temp_err)) :])
        )
        description["RMSE"] = "{:.0f}".format(
            np.sqrt(
                np.mean(
                    (
                        self.signal[max(-GRAPH_MEMORY, -len(self.signal)) :]
                        - self.consumption[max(-GRAPH_MEMORY, -len(self.consumption)) :]
                    )
                )
            )
        )
        description["Cumulative average offset"] = "{:.0f}".format(
            np.mean(
                self.signal[max(-GRAPH_MEMORY, -len(self.signal)) :]
                - self.consumption[max(-GRAPH_MEMORY, -len(self.consumption)) :]
            )
        )
        self.description.update({time_step: description})

    def update_houses_data(self, obs_dict: dict, time_step: int) -> None:
        houses_data = []
        for house_id in obs_dict.keys():
            houses_data.append({"id": house_id})
            if obs_dict[house_id]["turned_on"]:
                houses_data[house_id].update({"hvacStatus": "ON"})
            elif obs_dict[house_id]["lockout"]:
                houses_data[house_id].update(
                    {
                        "hvacStatus": "Lockout",
                        "secondsSinceOff": obs_dict[house_id]["seconds_since_off"],
                    }
                )
            else:
                houses_data[house_id].update(
                    {
                        "hvacStatus": "OFF",
                        "secondsSinceOff": obs_dict[house_id]["seconds_since_off"],
                    }
                )

            houses_data[house_id]["indoorTemp"] = obs_dict[house_id]["indoor_temp"]
            houses_data[house_id]["targetTemp"] = obs_dict[house_id]["target_temp"]
            houses_data[house_id]["tempDifference"] = (
                obs_dict[house_id]["indoor_temp"] - obs_dict[house_id]["target_temp"]
            )
        self.houses_data.update({time_step: houses_data})

    async def get_state_at(self, time_step: int) -> None:
        await self.socket_manager_service.emit("timeStepData", self.description[time_step])
        await self.socket_manager_service.emit("houseChange", self.houses_data[time_step])
                                               

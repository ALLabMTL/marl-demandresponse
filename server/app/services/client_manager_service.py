import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.services.socket_manager_service import SocketManager
from app.utils.logger import logger
from app.core.environment.environment import EnvironmentObsDict


DESCRIPTION_KEYS = [
    "Number of HVAC",
    "Number of locked HVAC",
    "Outdoor temperature",
    "Average indoor temperature",
    "Average temperature difference",
    "Regulation signal",
    "Current consumption",
    "Consumption error (%)",
    "RMSE",
]


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
    houses_data: Dict[int, EnvironmentObsDict]

    @property
    def description_values(self) -> list:
        values = [
            str(self.data_frame.shape[0]),
            str(
                np.where(
                    self.data_frame["lockout"],
                    1,
                    0,
                ).sum()
            ),
            str(round(self.data_frame["OD_temp"][0], 2)),
            str(round(self.data_frame["indoor_temp"].mean(), 2)),
            str(round(self.data_frame["temperature_difference"].mean(), 2)),
            str(self.data_frame["reg_signal"][0]),
            str(self.data_frame["cluster_hvac_power"][0]),
            str(
                (
                    self.data_frame["reg_signal"][0]
                    - self.data_frame["cluster_hvac_power"][0]
                )
                / self.data_frame["reg_signal"][0]
                * 100
            ),
            str(
                np.sqrt(
                    np.mean(
                        (
                            self.signal[-len(self.signal) :]
                            - self.consumption[-len(self.consumption) :]
                        )
                        ** 2
                    )
                )
            ),
            str(
                np.mean(
                    self.signal[-len(self.signal) :]
                    - self.consumption[-len(self.consumption) :]
                )
            ),
        ]
        return values

    def __init__(
        self,
        socket_manager_service: SocketManager,
    ) -> None:
        self.socket_manager = socket_manager_service

    def initialize_data(self, interface: bool) -> None:
        self.interface = interface
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

    async def update_data(
        self,
        obs_dict: Dict[int, EnvironmentObsDict],
        time_step: int,
    ) -> None:
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
        await self.log(
            emit=True, endpoint="houseChange", data=self.houses_data[time_step]
        )
        await self.log(
            emit=True,
            endpoint="dataChange",
            data=self.description[time_step],
        )

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

    def update_desc_data(self, time_step: int) -> None:
        description = dict(zip(DESCRIPTION_KEYS, self.description_values))
        self.description.update({time_step: description})

    def update_houses_data(
        self, obs_dict: Dict[int, EnvironmentObsDict], time_step: int
    ) -> None:
        houses_data = []

        for house_id, _ in enumerate(obs_dict):
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

    async def log(
        self,
        text: str = "",
        emit: bool = False,
        endpoint: str = "",
        data: Any = {},
    ) -> None:
        if self.interface and emit and endpoint != "":
            await self.socket_manager.emit(endpoint, data)
        if text != "":
            logger.info(text)

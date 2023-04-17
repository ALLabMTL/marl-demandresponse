from typing import Any, Dict

import numpy as np
import pandas as pd

from app.core.environment.environment import EnvironmentObsDict
from app.services.socket_manager_service import SocketManager
from app.utils.logger import logger

from .socket_manager_service import SocketManager

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
    "Mass temperature",
    "Target temperature",
    "Average temperature error",
]


class ClientManagerService:
    """
    Class that manages the client-side data and interactions with the server.

    Attributes:
        description (Dict[int, Dict[str, str]]): A dictionary containing a summary of the current state of the simulation.
        temp_diff (np.ndarray): An array containing the average temperature difference between the indoor and target temperatures.
        temp_err (np.ndarray): An array containing the average temperature error between the indoor and target temperatures.
        air_temp (np.ndarray): An array containing the average indoor temperature.
        mass_temp (np.ndarray): An array containing the average mass temperature.
        target_temp (np.ndarray): An array containing the average target temperature.
        outdoor_temp (np.ndarray): An array containing the average outdoor temperature.
        signal (np.ndarray): An array containing the current regulation signal.
        consumption (np.ndarray): An array containing the current consumption.
        recent_signal (np.ndarray): An array containing the recent regulation signal.
        recent_consumption (np.ndarray): An array containing the recent consumption.
        data_frame (pd.DataFrame): A dataframe containing the observation data for all houses.
        houses_data (Dict[int, EnvironmentObsDict]): A dictionary containing the observation data for each individual house.
    """
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
        """
        A property that returns a list of values to be used in the description of client data at each timestep.

        Returns:
        list: A list of values to be used in the description of client data at each timestep.
        """
        values = [
            str(self.data_frame.shape[0]),  # "Number of HVAC",
            str(
                np.where(
                    self.data_frame["lockout"],
                    1,
                    0,
                ).sum()
            ),  #  "Number of locked HVAC",
            str(round(self.data_frame["OD_temp"][0], 2)),  # "Outdoor temperature",
            str(
                round(self.data_frame["indoor_temp"].mean(), 2)
            ),  # "Average indoor temperature",
            str(
                round(self.data_frame["temperature_difference"].mean(), 2)
            ),  # "Average temperature difference",
            str(self.data_frame["reg_signal"][0]),  # "Regulation signal",
            str(self.data_frame["cluster_hvac_power"][0]),  # "Current consumption",
            str(
                (
                    self.data_frame["reg_signal"][0]
                    - self.data_frame["cluster_hvac_power"][0]
                )
                / self.data_frame["reg_signal"][0]
                * 100
            ),  # "Consumption error (%)",
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
            ),  # "RMSE",
            str(round(self.data_frame["mass_temp"][0], 2)),
            str(round(self.data_frame["target_temp"][0], 2)),
            str(np.mean(self.temp_err)),
        ]
        return values

    def __init__(
        self,
        socket_manager_service: SocketManager,
    ) -> None:
        """
        Initialize a new ClientManagerService object.

        Parameters:
            socket_manager_service (SocketManager): An instance of the SocketManager class.
        """
        self.socket_manager = socket_manager_service

    def initialize_data(self, interface: bool) -> None:
        """
        Initialize data attributes for the ClientManagerService object.

        Parameters:
            interface (bool): A boolean value indicating whether an interface is being used.
        """
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
        """
        Update data attributes for the ClientManagerService object.

        Parameters:
            obs_dict (Dict[int, EnvironmentObsDict]): A dictionary of dictionaries that stores environment data for each client at each timestep.
            time_step (int): An integer indicating the current timestep.
        """
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
        """Update graph data attributes for the ClientManagerService object."""
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
        """
        Update description data attributes for the ClientManagerService object.

        Parameters:
            time_step (int): An integer indicating the current timestep.
        """
        description = dict(zip(DESCRIPTION_KEYS, self.description_values))
        self.description.update({time_step: description})

    def update_houses_data(
        self, obs_dict: Dict[int, EnvironmentObsDict], time_step: int
    ) -> None:
        """
        Update houses data attributes for the ClientManagerService object.

        Parameters:
            obs_dict (Dict[int, EnvironmentObsDict]): A dictionary of dictionaries that stores environment data for each client at each timestep.
            time_step (int): An integer indicating the current timestep.
        """
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
        """
        Log a message and optionally emits it over a socket.

        Parameters:
            text (str): The message to be logged. If not provided, no message will be logged.
            emit (bool): Whether or not to emit the message over a socket. Defaults to False.
            endpoint (str): The endpoint to emit the message to. Only applicable if emit is True. Defaults to an empty string.
            data (Any): Any additional data to be included with the emitted message. Only applicable if emit is True. Defaults to an empty dictionary.

        Returns:
            None
        """

        if self.interface and emit and endpoint != "":
            await self.socket_manager.emit(endpoint, data)
        if text != "":
            logger.info(text)

    async def get_state_at(self, time_step: int) -> None:
        """
        Log the description and house data at a specific time step.

        Args:
            time_step (int): The time step to log the data for.

        Returns:
            None
        """
        await self.log(
            emit=True, endpoint="timeStepData", data=self.description[time_step]
        )
        await self.log(
            emit=True, endpoint="houseChange", data=self.houses_data[time_step]
        )

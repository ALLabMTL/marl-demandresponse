from copy import deepcopy
from typing import Dict
import numpy as np
import pandas as pd
from utils.logger import logger
from .socket_manager_service import SocketManager

GRAPH_MEMORY = 5000

class ClientManagerService():

    def __init__(self, socket_manager_service: SocketManager) -> None:
        self.socket_manager_service = socket_manager_service
        self.data_messages = {}
        self.temp_diff = np.array([])
        self.temp_err = np.array([])
        self.air_temp = np.array([])
        self.mass_temp = np.array([])
        self.target_temp = np.array([])
        self.OD_temp = np.array([])
        self.signal = np.array([])
        self.consumption = np.array([])
        self.houses_messages: Dict[int, dict] =  {}
    
    async def emit_data_change(self, obs_dict: Dict[int, dict]) -> pd.DataFrame:
        df = pd.DataFrame(obs_dict).transpose()
        self.update_data(df)

        for house_id in obs_dict.keys():
            self.houses_messages[house_id] = {}
            if obs_dict[house_id]["turned_on"]:
                self.houses_messages[house_id].update({"HVAC status" : "ON"})
            elif obs_dict[house_id]["lockout"]:
                self.houses_messages[house_id].update({"HVAC status" : "Lockout", "Seconds since off" : obs_dict[house_id]["seconds_since_off"]})
            else:
                self.houses_messages[house_id].update({"HVAC status" : "OFF", "Seconds since off" : obs_dict[house_id]["seconds_since_off"]})

            self.houses_messages[house_id]["Indoor temperature"] = obs_dict[house_id]["indoor_temp"]
            self.houses_messages[house_id]["Target temperature"] = obs_dict[house_id]["target_temp"]
            self.houses_messages[house_id]["Difference to target"] = obs_dict[house_id]["indoor_temp"] - obs_dict[house_id]["target_temp"]




        await self.socket_manager_service.emit("dataChange", self.data_messages)
        await self.socket_manager_service.emit("houseChange", self.houses_messages)


    def update_data(self, df: pd.DataFrame) -> None:
        df["temperature_difference"] = df["indoor_temp"] - df["target_temp"]
        df["temperature_error"] = np.abs(df["indoor_temp"] - df["target_temp"])
        self.temp_diff = np.append(self.temp_diff, df["temperature_difference"].mean())
        self.temp_err = np.append(self.temp_err, df["temperature_error"].mean())
        self.air_temp = np.append(self.air_temp, df["indoor_temp"].mean())
        self.mass_temp = np.append(self.mass_temp, df["mass_temp"].mean())
        self.target_temp = np.append(self.target_temp, df["target_temp"].mean())
        self.OD_temp = np.append(self.OD_temp, df["OD_temp"].mean())
        self.signal = np.append(self.signal, df["reg_signal"][0])
        self.consumption = np.append(self.consumption, df["cluster_hvac_power"][0])
        self.update_data_messages(df)

    def update_data_messages(self, df: pd.DataFrame):
        self.data_messages = {}
        self.data_messages["Number of HVAC"] = str(df.shape[0])
        self.data_messages["Number of locked HVAC"] = str(
            np.where(
                df["seconds_since_off"] > df["lockout_duration"], 1, 0
            ).sum()
        )
        self.data_messages["Outdoor temperature"] = (
            str(round(df["OD_temp"][0], 2)) + " °C"
        )
        self.data_messages["Average indoor temperature"] = (
            str(round(df["indoor_temp"].mean(), 2)) + " °C"
        )
        self.data_messages["Average temperature difference"] = (
            str(round(df["temperature_difference"].mean(), 2)) + " °C"
        )
        self.data_messages["Regulation signal"] = str(df["reg_signal"][0])
        self.data_messages["Current consumption"] = str(df["cluster_hvac_power"][0])
        self.data_messages["Consumption error (%)"] = "{:.3f}%".format(
            (df["reg_signal"][0] - df["cluster_hvac_power"][0])
            / df["reg_signal"][0]
            * 100
        )
        self.data_messages["RMSE"] = "{:.0f}".format(
            np.sqrt(
                np.mean(
                    (
                        self.signal[max(-GRAPH_MEMORY, -len(self.signal)) :]
                        - self.consumption[max(-GRAPH_MEMORY, -len(self.consumption)) :]
                    )
                    ** 2
                )
            )
        )
        self.data_messages["Cumulative average offset"] = "{:.0f}".format(
            np.mean(
                self.signal[max(-GRAPH_MEMORY, -len(self.signal)) :]
                - self.consumption[max(-GRAPH_MEMORY, -len(self.consumption)) :]
            )
        )


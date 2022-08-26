import sys

sys.path.append("../marl-demandresponse")
from numpy import roll
from .MPC import *
import pandas as pd
import time
from utils import house_solar_gain

global_mpc_memory = [None, None]


class MPCController(object):
    """MPC Controller"""

    def __init__(self, agent_properties, config_dict, num_state):
        self.time_step = 0
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

        self.id = self.id
        self.window_area = config_dict["default_house_prop"]["window_area"]
        self.shading_coeff = config_dict["default_house_prop"]["shading_coeff"]
        self.mode = config_dict["default_env_prop"]["power_grid_prop"]["signal_mode"]
        self.time_step_duration = config_dict["default_env_prop"]["time_step"]
        self.rolling_horizon = config_dict["MPC_prop"]["rolling_horizon"]
        if self.mode == "sinusoidals":
            self.signal_parameter = config_dict["default_env_prop"]["power_grid_prop"][
                "signal_parameters"
            ]["sinusoidals"]
        self.solar_gain = config_dict["default_house_prop"]["solar_gain_bool"]
        self.norm_signal_normalization = config_dict["default_env_prop"]["reward_prop"]["norm_reg_sig"]

    def act(self, obs):
        self.time_step += 1

        if self.time_step % 5 == 0:
            print("current_timestep : ", self.time_step)
        if global_mpc_memory[0] != self.time_step:
            start = time.time()
            df = pd.DataFrame(obs).transpose()
            nb_agents = len(df.index)
            Ua = df["house_Ua"].to_list()
            Ca = df["house_Ca"].to_list()
            Cm = df["house_Cm"].to_list()
            Hm = df["house_Hm"].to_list()
            initial_air_temperature = df["house_temp"].to_list()
            initial_mass_temperature = df["house_mass_temp"].to_list()
            target_temperature = df["house_target_temp"].to_list()
            remaining_lockout = (
                df["hvac_lockout_duration"] - df["hvac_seconds_since_off"]
            ) * df["hvac_turned_on"]
        
            rolling_horizon = self.rolling_horizon
            if self.solar_gain:
                solar_gain = [

                    house_solar_gain(
                        df["datetime"][0], self.window_area, self.shading_coeff
                    )
                ] * rolling_horizon
            else: 
                solar_gain = [
                    0
                ] * rolling_horizon
       
          
            time_step_duration = self.time_step_duration
            lockout_duration = df["hvac_lockout_duration"][0]
            reg_signal = [df["reg_signal"][0]] * rolling_horizon
            od_temp = [df["OD_temp"][0]] * rolling_horizon
            HVAC_consumption = df["hvac_cooling_capacity"] / df["hvac_COP"]
            HVAC_cooling = df["hvac_cooling_capacity"] / (
                1 + df["hvac_latent_cooling_fraction"]
            )
           
            global_mpc_memory[1] = best_MPC_action(
                nb_agents,
                HVAC_cooling,
                HVAC_consumption,
                initial_air_temperature,
                initial_mass_temperature,
                target_temperature,
                remaining_lockout,
                reg_signal,
                od_temp,
                solar_gain,
                Hm,
                Ca,
                Ua,
                Cm,
                rolling_horizon,
                time_step_duration,
                lockout_duration,
                self.norm_signal_normalization
            )
          
            global_mpc_memory[0] = self.time_step
           
        return global_mpc_memory[1][self.id]

import gym
import ray
import numpy as np
import warnings
import random

from datetime import datetime, timedelta
from datetime import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List, Any


def reg_signal_penalty(cluster_hvac_power, power_grid_reg_signal):
    penalty = (cluster_hvac_power - power_grid_reg_signal) ** 2
    return penalty


def compute_temp_penalty(target_temp, deadband, house_temp) -> float:
    """ Compute the temperature penalty for one house """
    if target_temp + deadband / 2 < house_temp:
        temperature_penalty = (house_temp - (target_temp + deadband / 2)) ** 2
    elif target_temp - deadband / 2 > house_temp:
        temperature_penalty = ((target_temp - deadband / 2) - house_temp) ** 2
    else:
        temperature_penalty = 0.0

    return temperature_penalty


class MADemandResponseEnv(MultiAgentEnv):
    """Multi agent demand response environment"""
    start_datetime: datetime
    datetime: datetime
    time_step: timedelta

    def __init__(self, env_properties):
        super(MADemandResponseEnv, self).__init__()

        datetime_format = "%Y-%m-%d %H:%M:%S"
        self.start_datetime = datetime.strptime(env_properties["start_datetime"],
                                                datetime_format)  # Start date and time (Y,M,D, H, min, s)
        self.datetime = self.start_datetime  # Start time in hour (24h format, decimal hours)
        self.time_step = timedelta(seconds=env_properties["time_step"])

        self.env_properties = env_properties
        self.agent_ids = env_properties["agent_ids"]

        self.cluster = ClusterHouses(env_properties["cluster_properties"], self.datetime, self.time_step)
        self.power_grid = PowerGrid(env_properties["power_grid_properties"], env_properties["nb_hvac"])

    def reset(self):
        self.datetime = self.start_datetime
        self.cluster = ClusterHouses(self.env_properties["cluster_properties"], self.datetime, self.time_step)
        cluster_obs_dict = self.cluster.make_obs_dict(self.datetime)

        obs_dict = cluster_obs_dict  # TODO: add powergrid
        return obs_dict

    def step(self, action_dict):
        self.datetime += self.time_step
        # Cluster step
        obs_dict, temp_penalty_dict, cluster_hvac_power, _ = self.cluster.step(self.datetime, action_dict,
                                                                               self.time_step)
        # Power grid step
        power_grid_reg_signal = self.power_grid.step(self.datetime)

        # Merge observations
        self.merge_cluster_powergrid_obs(obs_dict, power_grid_reg_signal, cluster_hvac_power)

        # Compute reward
        rewards_dict = self.compute_rewards(temp_penalty_dict, cluster_hvac_power, power_grid_reg_signal)
        dones_dict = self.make_done_dict()
        info_dict = {"cluster_hvac_power": cluster_hvac_power}
        #print("cluster_hvac_power: {}, power_grid_reg_signal: {}".format(cluster_hvac_power, power_grid_reg_signal))

        return obs_dict, rewards_dict, dones_dict, info_dict

    def merge_cluster_powergrid_obs(self, cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power) -> None:
        for agent_id in self.agent_ids:
            cluster_obs_dict[agent_id]["reg_signal"] = power_grid_reg_signal
            cluster_obs_dict[agent_id]["cluster_hvac_power"] = cluster_hvac_power

    def compute_rewards(self, temp_penalty_dict, cluster_hvac_power, power_grid_reg_signal):
        rewards_dict: dict[str, float] = {}
        signal_penalty = reg_signal_penalty(cluster_hvac_power, power_grid_reg_signal)
        for agent_id in self.agent_ids:
            rewards_dict[agent_id] = -1 * (temp_penalty_dict[agent_id] + self.env_properties["alpha"] * signal_penalty)
        return rewards_dict

    def make_done_dict(self):
        done_dict: dict[str, bool] = {}
        for agent_id in self.agent_ids:
            done_dict[agent_id] = False
        return done_dict


class HVAC(object):
    """ HVAC simulator """

    def __init__(self, hvac_properties, time_step):
        self.id = hvac_properties["id"]
        self.hvac_properties = hvac_properties
        self.COP = hvac_properties["COP"]  # Coefficient of performance (2.5)
        self.cooling_capacity = hvac_properties["cooling_capacity"]  # Cooling capacity (W)
        self.latent_cooling_fraction = hvac_properties[
            "latent_cooling_fraction"]  # Fraction of latent cooling w.r.t. sensible cooling
        self.lockout_duration = hvac_properties["lockout_duration"]  # Lockout duration (seconds)
        self.turned_on = False  # HVAC can be on (True) or off (False)
        self.seconds_since_off = self.lockout_duration  # Seconds since last turning off
        self.time_step = time_step

    def step(self, command):
        if command:  # command = on
            if self.turned_on:  # Ignore command
                self.seconds_since_off = 0  # Keep time counter at 0
            elif self.seconds_since_off >= self.lockout_duration:  # If lockout is over
                self.turned_on = True  # Turn on
                self.seconds_since_off = 0  # Keep time counter at 0
            else:  # Ignore command
                self.seconds_since_off += self.time_step.seconds  # Increment

        else:  # command = off
            if self.turned_on:
                self.turned_on = False  # Turn off
                self.seconds_since_off = 0  # Start time counter
            else:  # if already off
                self.seconds_since_off += self.time_step.seconds  # Increment time counter

        return self.turned_on

    def get_Q(self):
        if self.turned_on:
            q_hvac = -1 * self.cooling_capacity / (1 + self.latent_cooling_fraction)
        else:
            q_hvac = 0

        return q_hvac

    def power_consumption(self):
        if self.turned_on:
            return self.cooling_capacity / self.COP
        else:
            return 0


class SingleHouse(object):
    """ Single house simulator """

    def __init__(self, house_properties, time_step):

        """
        Initialize the house
        """
        self.id = house_properties["id"]  # Unique house ID
        self.init_temp = house_properties["init_temp"]  # Initial indoors air temperature (Celsius degrees)
        self.current_temp = self.init_temp  # Current indoors air temperature
        self.current_mass_temp = self.init_temp
        self.house_properties = house_properties  # To keep in memory

        # Thermal constraints
        self.target_temp = house_properties["target_temp"]  # Target indoors air temperature (Celsius degrees)
        self.deadband = house_properties[
            "deadband"]  # Deadband of tolerance around the target temperature (Celsius degrees)

        # Thermodynamic properties
        self.Ua = house_properties["Ua"]  # House conductance U_a ( )
        self.Cm = house_properties["Cm"]  # House mass Cm (kg)
        self.Ca = house_properties["Ca"]  # House air mass Ca (kg)
        self.Hm = house_properties["Hm"]  # Mass surface conductance Hm ( )

        # HVACs
        self.hvac_properties = house_properties["hvac_properties"]
        self.hvacs = {}
        self.hvacs_ids = []

        for hvac_prop in house_properties["hvac_properties"]:
            hvac = HVAC(hvac_prop, time_step)
            self.hvacs[hvac.id] = hvac
            self.hvacs_ids.append(hvac.id)

        self.disp_count = 0

    def step(self, od_temp, time_step):
        self.update_temperature(od_temp, time_step)

        self.disp_count += 1
        if self.disp_count >= 100:
            print("House ID: {} -- OD_temp : {:f}, ID_temp: {:f}, target_temp: {:f}, diff: {:f}, HVAC on: {}, HVAC lockdown: {}".format(
                self.id, od_temp, self.current_temp, self.target_temp, self.current_temp - self.target_temp,
                self.hvacs[self.id + "_1"].turned_on, self.hvacs[self.id + "_1"].seconds_since_off))
            self.disp_count = 0

    def update_temperature(self, od_temp, time_step):
        time_step_sec = time_step.seconds
        Hm, Ca, Ua, Cm = self.Hm, self.Ca, self.Ua, self.Cm

        # Temperatures in K
        od_temp_K = od_temp + 273
        current_temp_K = self.current_temp + 273
        current_mass_temp_K = self.current_mass_temp + 273

        # Model taken from http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide

        # Heat from hvacs (negative if it is AC)
        total_Qhvac = 0
        for hvac_id in self.hvacs_ids:
            hvac = self.hvacs[hvac_id]
            total_Qhvac += hvac.get_Q()

        # Total heat addition to air
        other_Qa = 0  # windows, ...
        Qa = total_Qhvac + other_Qa
        # Heat from inside devices (oven, windows, etc)
        Qm = 0

        # Variables and constants
        a = Cm * Ca / Hm
        b = Cm * (Ua + Hm) / Hm + Ca
        c = Ua
        d = Qm + Qa + Ua * od_temp_K
        g = Qm / Hm

        r1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        r2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        dTA0dt = Hm / (Ca * current_mass_temp_K) - (Ua + Hm) / (Ca * current_temp_K) + Ua / (
                Ca * od_temp_K) + Qa / Ca

        A1 = (r2 * current_temp_K - dTA0dt - r2 * d / c) / (r2 - r1)
        A2 = current_temp_K - d / c - A1
        A3 = r1 * Ca / Hm + (Ua + Hm) / Hm
        A4 = r2 * Ca / Hm + (Ua + Hm) / Hm

        # Updating the temperature
        old_temp_K = current_temp_K
        new_current_temp_K = A1 * np.exp(r1 * time_step_sec) + A2 * np.exp(r2 * time_step_sec) + d / c
        new_current_mass_temp_K = A1 * A3 * np.exp(r1 * time_step_sec) + A2 * A4 * np.exp(r2 * time_step_sec) + g + d / c


        self.current_temp = new_current_temp_K - 273   
        self.current_mass_temp = new_current_mass_temp_K - 273

        #if np.abs(old_temp_K - current_temp_K) > 1 or True:
        #    print("Old ID temp: {}, current ID temp: {}, time_step_sec: {}".format(old_temp_K, current_temp_K, time_step_sec))
        #    print("A1: {}, r1: {}, A2: {}, r2: {}, d: {}, c: {}, dTA0dt: {}".format(A1, r1, A2, r2, d, c, dTA0dt))


class ClusterHouses(object):
    """ A cluster contains several houses, has the same outdoors temperature, and has one tracking signal """

    def __init__(self, cluster_properties, date_time, time_step):
        """
        Initialize the cluster of houses
        """
        self.cluster_properties = cluster_properties

        # Houses
        self.houses = {}
        self.hvacs_id_registry = {}  # Registry mapping each hvac_id to its house id
        for house_properties in cluster_properties["houses_properties"]:
            house = SingleHouse(house_properties, time_step)
            self.houses[house.id] = house
            for hvac_id in house.hvacs_ids:
                self.hvacs_id_registry[hvac_id] = house.id

        # Outdoors temperature profile
        ## Currently modeled as noisy sinusoidal
        self.day_temp = cluster_properties["day_temp"]
        self.night_temp = cluster_properties["night_temp"]
        self.temp_std = cluster_properties["temp_std"]  # Std-dev of the white noise applied on outdoors temperature
        self.current_OD_temp = self.compute_OD_temp(date_time)

    def make_obs_dict(self, date_time):
        obs_dictionary = {}
        for hvac_id in self.hvacs_id_registry.keys():
            obs_dictionary[hvac_id] = {}

            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]

            # Dynamic values from cluster
            obs_dictionary[hvac_id]["OD_temp"] = self.current_OD_temp
            obs_dictionary[hvac_id]["datetime"] = date_time

            # Dynamic values from house
            obs_dictionary[hvac_id]["house_temp"] = house.current_temp

            # Dynamic values from HVAC
            obs_dictionary[hvac_id]["hvac_turned_on"] = hvac.turned_on
            obs_dictionary[hvac_id]["hvac_seconds_since_off"] = hvac.seconds_since_off

            # Supposedly constant values from house (may be changed later)
            obs_dictionary[hvac_id]["house_target_temp"] = house.target_temp
            obs_dictionary[hvac_id]["house_deadband"] = house.deadband
            obs_dictionary[hvac_id]["house_Ua"] = house.Ua
            obs_dictionary[hvac_id]["house_Cm"] = house.Cm
            obs_dictionary[hvac_id]["house_Ca"] = house.Ca
            obs_dictionary[hvac_id]["house_Hm"] = house.Hm

            # Supposedly constant values from hvac
            obs_dictionary[hvac_id]["hvac_COP"] = hvac.COP
            obs_dictionary[hvac_id]["hvac_cooling_capacity"] = hvac.cooling_capacity
            obs_dictionary[hvac_id]["hvac_latent_cooling_fraction"] = hvac.latent_cooling_fraction
            obs_dictionary[hvac_id]["hvac_lockout_duration"] = hvac.lockout_duration

        return obs_dictionary

    def step(self, date_time, actions_dict, time_step):
        ## Enact actions

        # Send commend to the hvacs
        for hvac_id in self.hvacs_id_registry.keys():
            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]
            if hvac_id in actions_dict.keys():
                command = actions_dict[hvac_id]
            else:
                warnings.warn("HVAC {} in house {} did not receive any command.".format(hvac_id, house_id))
                command = False
            hvac.step(command)

        # Update outdoors temperature
        self.current_OD_temp = self.compute_OD_temp(date_time)

        # Update houses' temperatures
        for house_id in self.houses.keys():
            house = self.houses[house_id]
            house.step(self.current_OD_temp, time_step)

        ## Observations
        obs_dictionary = self.make_obs_dict(date_time)

        ## Temperature penalties and total cluster power consumption
        temp_penalty_dict = {}
        cluster_hvac_power = 0

        for hvac_id in self.hvacs_id_registry.keys():
            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]

            # Temperature penalties
            temp_penalty_dict[hvac.id] = compute_temp_penalty(house.target_temp, house.deadband, house.current_temp)

            # Cluster hvac power consumption
            cluster_hvac_power += hvac.power_consumption()

        # Info
        info_dict = {}  # TODO

        return obs_dictionary, temp_penalty_dict, cluster_hvac_power, info_dict

    def compute_OD_temp(self, date_time) -> float:
        """ Compute the outdoors temperature based on the time, according to a noisy sinusoidal model"""
        amplitude = (self.day_temp - self.night_temp) / 2
        bias = (self.day_temp + self.night_temp) / 2
        delay = -6  # Temperature is coldest at 6am
        time_day = date_time.hour + date_time.minute / 60.0

        temperature = amplitude * np.sin(2 * np.pi * (time_day + delay) / 24) + bias

        temperature += random.gauss(0, self.temp_std)
        # TODO : add noise

        return temperature


class PowerGrid(object):
    def __init__(self, power_grid_properties, nb_hvacs):
        self.avg_power_per_hvac = power_grid_properties["avg_power_per_hvac"]
        self.noise_mode = power_grid_properties["noise_mode"]
        self.noise_params = power_grid_properties["noise_parameters"][self.noise_mode]
        self.nb_hvac = nb_hvacs
        self.init_signal = power_grid_properties["init_signal"]
        self.current_signal = self.init_signal

    def step(self, date_time) -> float:
        if self.noise_mode == "none":
            pass
        elif self.noise_mode == "sinusoidal":
            """ Compute the outdoors temperature based on the time, according to a noisy sinusoidal model"""
            amplitude = self.noise_params["amplitude_per_hvac"] * self.nb_hvac
            wavelength = self.noise_params["wavelength"]
            bias = self.avg_power_per_hvac * self.nb_hvac
            time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second
            self.current_signal = amplitude * np.sin(2 * np.pi * time_sec / wavelength) + bias
        return self.current_signal

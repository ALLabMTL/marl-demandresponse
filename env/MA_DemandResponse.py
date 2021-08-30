import gym
import ray
import pandas as pd
from datetime import datetime, timedelta
from datetime import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List

class MA_DemandResponseEnv(MultiAgentEnv):
	"""Multi agent demand response environment"""
	def __init__(self, env_properties):
		super(MA_DemandResponseEnv, self).__init__()

		datetime_format = "%Y-%m-%d %H:%M:%S"
		self.start_datetime = datetime.strptime(env_properties["start_datetime"], datetime_format) 	# Start date and time (Y,M,D, H, min, s)
		self.datetime = self.start_datetime		# Start time in hour (24h format, decimal hours)
		self.time_step = timedelta(seconds = env_properties["time_step"])
		self.properties = env_properties
		self.cluster = ClusterHouses(env_properties["cluster_properties"])
		#self.power_grid = PowerGrid(power_grid_properties)

	def reset(self):
		obs_dict = {}
		for agent in self.agent_keys:
			obs_dict[agent] = [0, 1] 		# Place-holder for agent observations
		return obs_dict


	def step(self, action_dict):
		pass
		#obs_dict = {}
		#reward_dict = {}
		#dones_dict = {}
		#info_dict = {}
		#for agent in self.agent_keys:
	#		if agent in action_dict.keys():
	#			action = action_dict[agent]
	#		else:
	#			action = 0
	#		obs_dict[agent] = [action] 		# Place-holder for agent observations
	#		reward_dict[agent] = 1 			# Place-holder for agent reward
	#		dones_dict[agent] = False 		# Place-holder for agent done
	#		info_dict[agent] = []			# Place-holder for optional info value


	#	return obs_dict, reward_dict, dones_dict, info_dict

class HVAC(object):
	""" HVAC simulator """
	def __init__(self, hvac_properties):
		self.id = hvac_properties["id"]
		self.COP = hvac_properties["COP"]										# Coefficient of performance (2.5)
		self.cooling_capacity = hvac_properties["cooling_capacity"]    			# Cooling capacity (1)
		self.sensible_cooling_ratio = hvac_properties["sensible_cooling_ratio"] # Ratio of sensible cooling/total cooling (vs latent cooling)
		self.nominal_power = hvac_properties["nominal_power"] 					# Power when on (W) (10)
		self.lockout_duration = hvac_properties["lockout_duration"] 			# Lockout duration (time steps) (5)
		self.turned_on = False  												# HVAC can be on (True) or off (False)
		self.steps_since_off = self.lockout_duration 							# Seconds since last turning off

	def step(self, command):
		if command == True:
			if self.steps_since_off >= self.lockout_duration: # Turn on if possible
				self.turned_on = True
				self.steps_since_off = 0
			else:
				self.turned_on = self.turned_on		# Otherwise, ignore command

		else:
			if self.turned_on:
				self.steps_since_off = 0  		# Start time counter
			else:
				self.steps_since_off += 1			# Increment time counter
			self.turned_on = False					# Turn off

		if self.turned_on:
			Q_hvac = self.nominal_power / (self.COP*self.sensible_cooling_ratio/self.capacity)
		else:
			Q_hvac = 0
			
		return Q_hvac


class SingleHouse(object):
	""" Single house simulator """
	def __init__(self, house_properties):

		"""
		Initialize the house
		"""
		self.id = house_properties["id"] 	# Unique house ID
		self.init_temp = house_properties["init_temp"]  # Initial temperature (Celsius degrees)
		self.house_properties = house_properties 		# To keep in memory

		# Thermal constraints
		self.target_temp = house_properties["target_temp"] 	# Target temperature (Celsius degrees)
		self.deadband = house_properties["deadband"]		# Deadband of tolerance around the target temperature (Celsius degrees)

		# Thermodynamic properties
		self.Ua = house_properties["Ua"]  			# House conductance U_a ( )
		self.Cm = house_properties["Cm"] 			# House mass Cm (kg)
		self.Ca = house_properties["Ca"]			# House air mass Ca (kg)
		self.Hm = house_properties["Hm"]			# Mass surface conductance Hm ( )
		
		# HVACs
		self.hvac_properties = house_properties["hvac_properties"]
		self.hvacs = []
		for hvac_prop in house_properties["hvac_properties"]:
			hvac = HVAC(hvac_prop)
			self.hvacs.append(hvac)

	def step(self, action):
		pass



class ClusterHouses(object):
	""" A cluster contains several houses, has the same outdoors temperature, and has one tracking signal """
	def __init__(self, cluster_properties):
		"""
		Initialize the cluster of houses
		"""
		self.cluster_properties = cluster_properties

		# Houses
		houses = []
		for house_properties in cluster_properties["houses_properties"]:
			house = SingleHouse(house_properties)
			houses.append(house)

		# Outdoors temperature profile
			## Currently modeled as noisy sinusoidal
		self.day_temp = cluster_properties["day_temp"]
		self.night_temp = cluster_properties["night_temp"]
		self.temp_std = cluster_properties["temp_std"]  	# Std-dev of the white noise applied on outdoors temperature



	def step(self, time, action):
		pass
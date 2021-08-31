import gym
import ray
import numpy as np
import warnings

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
		self.env_properties = env_properties
		#self.cluster = ClusterHouses(env_properties["cluster_properties"], self.datetime)
		#self.power_grid = PowerGrid(power_grid_properties)

	def reset(self):
		self.datetime = self.start_datetime
		self.cluster = ClusterHouses(self.env_properties["cluster_properties"], self.datetime)
		cluster_obs_dict = self.cluster.obsDict(self.datetime)

		obs_dict = cluster_obs_dict # TODO: add powergrid
		return obs_dict 


	def step(self, action_dict):
		self.datetime += self.time_step


		cl_obs_dict, temp_penalty_dict, cluster_hvac_power, _ = self.cluster.step(self.datetime, action_dict, self.time_step)


		obs_dict = {}
		reward_dict = {}
		dones_dict = {}
		info_dict = {}
		

		return obs_dict, reward_dict, dones_dict, info_dict

class HVAC(object):
	""" HVAC simulator """
	def __init__(self, hvac_properties):
		self.id = hvac_properties["id"]
		self.hvac_properties = hvac_properties
		self.COP = hvac_properties["COP"]										# Coefficient of performance (2.5)
		self.cooling_capacity = hvac_properties["cooling_capacity"]    			# Cooling capacity (W)
		self.latent_cooling_fraction = hvac_properties["latent_cooling_fraction"] # Fraction of latent cooling w.r.t. sensible cooling
		self.lockout_duration = hvac_properties["lockout_duration"] 			# Lockout duration (time steps)
		self.turned_on = False  												# HVAC can be on (True) or off (False)
		self.steps_since_off = self.lockout_duration 							# Seconds since last turning off


	def step(self, command):
		if command == True:
			if self.steps_since_off >= self.lockout_duration: 
				self.turned_on = True 				# Turn on
				self.steps_since_off += 1			# Increment time counter

			else:
				self.turned_on = self.turned_on		# Ignore command
				self.steps_since_off += 1			# Increment time counter


		else: 	# command = off
			if self.turned_on:						# if turning off
				self.steps_since_off = 0  			# Start time counter
			else:	# if already off
				self.steps_since_off += 1			# Increment time counter
			self.turned_on = False					# Turn off

		return self.turned_on

	def getQ(self):
		if self.turned_on:
			Q_hvac = -1*self.cooling_capacity/(1+self.latent_cooling_fraction)
		else:
			Q_hvac = 0

		return Q_hvac

	def powerConsumption(self):
		if self.turned_on:
			return self.cooling_capacity/self.COP
		else:
			return 0



class SingleHouse(object):
	""" Single house simulator """
	def __init__(self, house_properties):

		"""
		Initialize the house
		"""
		self.id = house_properties["id"] 	# Unique house ID
		self.init_temp = house_properties["init_temp"]  # Initial indoors air temperature (Celsius degrees)
		self.current_temp = self.init_temp				# Current indoors air temperature
		self.current_mass_temp = self.init_temp
		self.house_properties = house_properties 		# To keep in memory

		# Thermal constraints
		self.target_temp = house_properties["target_temp"] 	# Target indoors air temperature (Celsius degrees)
		self.deadband = house_properties["deadband"]		# Deadband of tolerance around the target temperature (Celsius degrees)

		# Thermodynamic properties
		self.Ua = house_properties["Ua"]  			# House conductance U_a ( )
		self.Cm = house_properties["Cm"] 			# House mass Cm (kg)
		self.Ca = house_properties["Ca"]			# House air mass Ca (kg)
		self.Hm = house_properties["Hm"]			# Mass surface conductance Hm ( )
		
		# HVACs
		self.hvac_properties = house_properties["hvac_properties"]
		self.hvacs = {}
		self.hvacs_ids = []

		for hvac_prop in house_properties["hvac_properties"]:
			hvac = HVAC(hvac_prop)
			self.hvacs[hvac.id] = hvac
			self.hvacs_ids.append(hvac.id)

	def step(self, OD_temp, time_step):
		self.updateTemperature(OD_temp, time_step)
		print("House ID: {} -- OD_temp : {}, ID_temp: {}".format(self.id, OD_temp, self.current_temp))

	def updateTemperature(self, OD_temp, time_step):
		time_step_sec = time_step.seconds
		Hm, Ca, Ua, Cm = self.Hm, self.Ca, self.Ua, self.Cm
		
		# Model taken from http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide

		# Heat addition from hvacs (negative if it is AC)
		total_Qhvac = 0
		for hvac_id in self.hvacs_ids:
			hvac = self.hvacs[hvac_id]
			total_Qhvac += hvac.getQ()

		# Total heat addition to air 
		otherQa = 0  					# windows, ...
		Qa = total_Qhvac + otherQa
		# Heat addition from inside devices (oven, windows, etc)
		Qm = 0

		# Variables and constants
		a = Cm*Ca/Hm
		b = Cm*(Ua + Hm)/Hm + Ca
		c = Ua
		d = Qm + Qa + Ua*OD_temp
		g = Qm/Hm

		r1 = (-b + np.sqrt(b**2 -4*a*c))/(2*a)
		r2 = (-b - np.sqrt(b**2 -4*a*c))/(2*a)


		dTA0dt = Hm/(Ca * self.current_mass_temp) - (Ua+Hm)/(Ca*self.current_temp) + Ua/(Ca*OD_temp) + Qa/Ca

		A1 = (r2*self.current_temp - dTA0dt - r2*d/c)/(r2-r1)
		A2 = self.current_temp - d/c - A1
		A3 = r1*Ca/Hm + (Ua+Hm)/Hm
		A4 = r2*Ca/Hm + (Ua+Hm)/Hm

		# Updating the temperature
		self.current_temp = A1*np.exp(r1*time_step_sec) + A2*np.exp(r2*time_step_sec) + d/c
		self.current_mass_temp = A1*A3*np.exp(r1*time_step_sec) + A2*A4*np.exp(r2*time_step_sec) + g + d/c



class ClusterHouses(object):
	""" A cluster contains several houses, has the same outdoors temperature, and has one tracking signal """
	def __init__(self, cluster_properties, datetime):
		"""
		Initialize the cluster of houses
		"""
		self.cluster_properties = cluster_properties

		# Houses
		self.houses = {}
		self.hvacs_id_registry = {} 			# Registry mapping each hvac_id to its house id
		for house_properties in cluster_properties["houses_properties"]:
			house = SingleHouse(house_properties)
			self.houses[house.id] = house
			for hvac_id in house.hvacs_ids:
				self.hvacs_id_registry[hvac_id] = house.id


		# Outdoors temperature profile
			## Currently modeled as noisy sinusoidal
		self.day_temp = cluster_properties["day_temp"]
		self.night_temp = cluster_properties["night_temp"]
		self.temp_std = cluster_properties["temp_std"]  	# Std-dev of the white noise applied on outdoors temperature
		self.current_OD_temp = self.computeODTemp(datetime)

	def obsDict(self, datetime):
		obs_dictionary = {}
		for hvac_id in self.hvacs_id_registry.keys():
			obs_dictionary[hvac_id] = {}

			# Getting the house and the HVAC
			house_id = self.hvacs_id_registry[hvac_id]
			house = self.houses[house_id]
			hvac = house.hvacs[hvac_id]

			# Dynamic values from cluster
			obs_dictionary[hvac_id]["OD_temp"] = self.current_OD_temp
			obs_dictionary[hvac_id]["datetime"] = datetime
			

			# Dynamic values from house
			obs_dictionary[hvac_id]["house_temp"] = house.current_temp
			
			# Dynamic values from HVAC
			obs_dictionary[hvac_id]["hvac_turned_on"] = hvac.turned_on
			obs_dictionary[hvac_id]["hvac_steps_since_off"] = hvac.steps_since_off

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


	def step(self, datetime, actions_dict, time_step):
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
		self.current_OD_temp = self.computeODTemp(datetime)


		# Update houses' temperatures
		for house_id in self.houses.keys():
			house = self.houses[house_id]
			house.step(self.current_OD_temp, time_step)



		## Observations
		obs_dictionary = self.obsDict(datetime)

		## Temperature penalties and total cluster power consumption
		temp_penalty_dict = {}
		cluster_hvac_power = 0

		for hvac_id in self.hvacs_id_registry.keys():
			# Getting the house and the HVAC
			house_id = self.hvacs_id_registry[hvac_id]
			house = self.houses[house_id]
			hvac = house.hvacs[hvac_id]

			# Temperature penalties
			temp_penalty_dict[hvac.id] = self.computeTempPenalty(house.target_temp, house.deadband, house.current_temp)

			# Cluster hvac power consumption
			cluster_hvac_power += hvac.powerConsumption()

		# Info
		info_dict = {} # TODO

		return obs_dictionary, temp_penalty_dict, cluster_hvac_power, info_dict


	def computeODTemp(self, time):
		""" Compute the outdoors temperature based on the time, according to a noisy sinusoidal model"""
		amplitude = (self.day_temp - self.night_temp)/2
		bias = (self.day_temp + self.night_temp)/2
		delay = -6 											# Temperature is coldest at 6am
		time_day = time.hour + time.minute/60.0

		temperature = amplitude * np.sin(2*np.pi*(time_day + delay)/24) + bias 
		# TODO : add noise

		return temperature

	def computeTempPenalty(self, target_temp, deadband, house_temp):
		""" Compute the temperature penalty for one house """
		if target_temp + deadband/2 < house_temp :
			temperature_penalty = (house_temp - (target_temp + deadband/2))**2
		elif target_temp - deadband/2 > house_temp :
			temperature_penalty = ((target_temp - deadband/2) - house_temp)**2
		else:
			temperature_penalty = 0

		return temperature_penalty



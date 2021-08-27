import gym
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List

class MA_DemandResponseEnv(MultiAgentEnv):
	"""Multi agent demand response environment"""
	def __init__(self):
		super(MA_DemandResponseEnv, self).__init__()
		self.agent_keys = ["house0", "house1","house2"]

		
	def reset(self):
		obs_dict = {}
		for agent in self.agent_keys:
			obs_dict[agent] = [0, 1] 		# Place-holder for agent observations
		return obs_dict


	def step(self, action_dict):
		obs_dict = {}
		reward_dict = {}
		dones_dict = {}
		info_dict = {}
		for agent in self.agent_keys:
			if agent in action_dict.keys():
				action = action_dict[agent]
			else:
				action = 0
			obs_dict[agent] = [action] 		# Place-holder for agent observations
			reward_dict[agent] = 1 			# Place-holder for agent reward
			dones_dict[agent] = False 		# Place-holder for agent done
			info_dict[agent] = []			# Place-holder for optional info value


		return obs_dict, reward_dict, dones_dict, info_dict

class HVAC(object)
	""" HVAC simulator """
	def __init__(self):
		self.COP = 2.5 			# Coefficient of performance
		self.cooling_cap = 1    # Cooling capacity
		self.sensible_cooling_ratio = 0.75 # Ratio of sensible cooling/total cooling (vs latent cooling)
		self.nominal_power = 10 # Power when on (W)
		self.lockout_duration = 5 # Lockout duration (seconds)
		self.turned_on = False  	# Status can be on or off
		self.seconds_since_change = self.lockout_duration # Seconds since last turning off

	def step(self, command):
		if command = True:
			if self.seconds_since_off >= self.lockout_duration: # Turn on if possible
				self.turned_on = True
				self.seconds_since_off = 0
			else:
				self.turned_on = self.turned_on		# Otherwise, ignore command

		else:
			if self.turned_on:
				self.seconds_since_off = 0  		# Start time counter
			else:
				self.seconds_since_off += 1			# Increment time counter
			self.turned_on = False					# Turn off

		if self.turned_on:
			Q_hvac = self.nominal_power / (self.COP*self.sensible_cooling_ratio/self.capacity)
		else:
			Q_hvac = 0
			
		return Q_hvac


class SingleHouse(object):
	""" Single house simulator """
	def __init__(self):
		self.Ua = 5  			#Place-holder for house conductance U_a ( )
		self.Cm = 10000 		# Placeholder for house mass Cm (kg)
		self.Ca = 20			# Placeholder for house air mass Ca (kg)
		self.Hm = 1				# Placeholder for mass surface conductance Hm ( )
		self.hvac = HVAC()

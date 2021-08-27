import gym
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List

class MA_DemandResponseEnv(MultiAgentEnv):
	"""docstring for ClassName"""
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



print("Hello World")

env = MA_DemandResponseEnv()
action_dict = {
	"house0": 0,
	"house1": 1,
	"house2": 2}
a = env.reset()
print(a)
b = env.step(action_dict)
print(b)

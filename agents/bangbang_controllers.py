




class AlwaysOnController(object):
	""" Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

	def __init__(self, agent_properties, config_dict):
		self.agent_properties = agent_properties
		self.id = agent_properties["id"]

	def act(self,obs):
		return True



class DeadbandBangBangController(object):
	""" Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

	def __init__(self, agent_properties, config_dict):
		self.agent_properties = agent_properties
		self.id = agent_properties["id"]

	def act(self,obs):
		house_temp = obs["house_temp"]
		house_target_temp = obs["house_target_temp"]
		house_deadband = obs["house_deadband"]
		hvac_turned_on = obs["hvac_turned_on"]


		if house_temp < house_target_temp-house_deadband/2:
			action = False
			#print("Too cold!")

		elif house_temp > house_target_temp+house_deadband/2:
			action = True
			#print("Too hot!")
		else:
			action = hvac_turned_on

		return action

class BangBangController(object):
	""" 
	Cools when temperature is hotter than target (no interest for deadband). Limited on the hardware side by lockout (but does not know about it)
	"""

	def __init__(self, agent_properties, config_dict):
		self.agent_properties = agent_properties
		self.id = agent_properties["id"]

	def act(self,obs):
		house_temp = obs["house_temp"]
		house_target_temp = obs["house_target_temp"]


		if house_temp <= house_target_temp:
			action = False

		elif house_temp > house_target_temp:
			action = True

		return action




class BasicController(object):
	""" Not really a bang bang controller but: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

	def __init__(self, agent_properties, config_dict):
		self.agent_properties = agent_properties
		self.id = agent_properties["id"]

	def act(self,obs):
		house_temp = obs["house_temp"]
		house_target_temp = obs["house_target_temp"]
		house_deadband = obs["house_deadband"]
		hvac_turned_on = obs["hvac_turned_on"]


		if house_temp < house_target_temp-house_deadband/2:
			action = False
			#print("Too cold!")

		elif house_temp > house_target_temp+house_deadband/2:
			action = True
			#print("Too hot!")
		else:
			action = hvac_turned_on

		return action


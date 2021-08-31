from env import *
from agents import *

from copy import deepcopy
import warnings



nb_houses = 1

# Default properties:

default_house_prop = {
	"id": 1,
	"init_temp": 20,
	"target_temp": 20,
	"deadband": 2,
	"Ua" : 84,									# House walls conductance (W/K) (default: 84W/K = 522 Btu/F/hr)
	"Cm" : 100*40700,							# House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)
	"Ca" : 100*2.5*1200,						# Air thermal mass in the house (J/K) (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)
	"Hm" : 455*8.29,							# House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.29 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))
}

default_hvac_prop = {
	"id": 1,
	"COP": 2.5,									# Coefficient of performance (power spent vs heat displaced)
	"cooling_capacity": 84*(30-20),				# Cooling capacity (W) (by design, Ua * (max OD temp - target ID temp))
	"latent_cooling_fraction": 0.35,			# Fraction of latent cooling w.r.t. sensible cooling
	"lockout_duration": 1						# In number of steps (TODO: change to seconds)
}

# Creating houses
houses_properties = []
for i in range(nb_houses):
	house_prop = deepcopy(default_house_prop)
	house_prop["id"] = str(i) 
	hvac_prop = deepcopy(default_hvac_prop)
	hvac_prop["id"] = str(i) + "_1"
	house_prop["hvac_properties"] = [hvac_prop]
	houses_properties.append(house_prop)




# Setting environment properties
env_properties = {
	"start_datetime": '2021-01-01 06:00:00',   	# Start date and time (Y-m-d H:M:S)
	"time_step": 2000,							# Time step in seconds
	"cluster_properties": {
		"day_temp": 30,							# Day temperature
		"night_temp": 23,						# Night temperature
		"temp_std": 0.5,						# Noise std dev on the temperature
		"houses_properties": houses_properties
	}

}




def getActions(actors, obs_dict):
	actions = {}
	for agent_id in actors.keys():
		actions[agent_id] = actors[agent_id].act(obs_dict[agent_id])
	return actions





env = MA_DemandResponseEnv(env_properties)
hvacs_id_registry = env.cluster.hvacs_id_registry

actors = {}
for hvac_id in hvacs_id_registry.keys():
	agent_prop = {}
	agent_prop["id"] = hvac_id
	actors[hvac_id] = BangBangController(agent_prop)


obs = env.reset()

for i in range(200):
	actions = getActions(actors, obs)
	obs, _, _, _ = env.step(actions)






# Testing environment
#action_dict = {
#	"house0": 0,
#	"house1": 1,
#	"house2": 2}
#obs = env.reset()
#print(obs)
#obs, rewards, dones, info  = env.step(action_dict)
#print([obs, rewards, dones, info])

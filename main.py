from env import *





nb_houses = 10

# Default properties:

default_hvac_prop = {
	"id": 1,
	"COP": 2.5,									# Coefficient of performance (power spent vs heat displaced)
	"cooling_capacity": 1,
	"sensible_cooling_ratio": 0.75,				# Ratio of sensible cooling/total cooling (vs latent cooling)
	"nominal_power": 10,						# In Watts
	"lockout_duration": 1						# In number of steps (TODO: change to seconds)
}

default_house_prop = {
	"id": 1,
	"init_temp": 20,
	"target_temp": 20,
	"deadband": 2,
	"Ua" : 1,									# House walls conductance (W/K)
	"Cm" : 100000,								# House mass (kg)
	"Ca" : 100,									# Air mass in the house (kg)
	"Hm" : 2,									# House mass surface conductance (W/K)
}

# Creating houses
houses_properties = []
for i in range(nb_houses):
	house_prop = default_house_prop
	house_prop["id"] = str(i) 
	hvac_prop = default_hvac_prop
	hvac_prop["id"] = str(i) + "_1"
	house_prop["hvac_properties"] = [hvac_prop]
	houses_properties.append(house_prop)




# Setting environment properties
env_properties = {
	"start_datetime": '2021-01-01 06:00:00',   	# Start date and time (Y-m-d H:M:S)
	"time_step": 20,							# Time step in seconds
	"cluster_properties": {
		"day_temp": 30,							# Day temperature
		"night_temp": 23,						# Night temperature
		"temp_std": 0.5,						# Noise std dev on the temperature
		"houses_properties": houses_properties
	}

}

print(env_properties)

print(env_properties["cluster_properties"]["houses_properties"][0]["hvac_properties"])


env = MA_DemandResponseEnv(env_properties)






# Testing environment
#action_dict = {
#	"house0": 0,
#	"house1": 1,
#	"house2": 2}
#obs = env.reset()
#print(obs)
#obs, rewards, dones, info  = env.step(action_dict)
#print([obs, rewards, dones, info])

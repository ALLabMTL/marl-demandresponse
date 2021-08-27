from env import *




env = MA_DemandResponseEnv()

# Testing environment
action_dict = {
	"house0": 0,
	"house1": 1,
	"house2": 2}
obs = env.reset()
print(obs)
obs, rewards, dones, info  = env.step(action_dict)
print([obs, rewards, dones, info])

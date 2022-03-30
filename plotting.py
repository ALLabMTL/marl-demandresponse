#%% Imports

import numpy as np
import random
import matplotlib.pyplot as plt
from utils import normStateDict

#%% Functions

def plot_env_test(env, action_type='off', n_steps=1000):
    assert action_type in ['off', 'on', 'random'], 'Action types available: off/on/random' 
    action_types = {'on': 1, 'off': 0, 'random': 0}
    
    # Reset environment
    obs_dict = env.reset()
    
    # Initialize arrays
    reward = np.empty(n_steps)
    hvac = np.empty(n_steps)
    temp = np.empty(n_steps)
    
    # Act on environment and save reward, hvac status and temperature
    for t in range(n_steps):
        if action_type == 'random':
            action = {"0_1": random.randint(0,1)}
        else:
            action = {"0_1": action_types[action_type]}
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        
        # Save data in arrays
        reward[t] = rewards_dict["0_1"]
        hvac[t] = next_obs_dict["0_1"]["hvac_turned_on"]
        temp[t] = next_obs_dict["0_1"]["house_temp"]

    plt.scatter(np.arange(len(hvac)), hvac, s=1, marker='.', c='orange')
    plt.plot(reward)
    plt.title('HVAC state vs. Reward')
    plt.show()
    plt.plot(temp)
    plt.title('Temperature')
    plt.show()
        
def plot_agent_test(env, agent, config_dict, n_steps=1000):      
    # Reset environment
    obs_dict = env.reset()
    cumul_avg_reward = 0
    
    # Initialize arrays
    reward = np.empty(n_steps)
    hvac = np.empty(n_steps)
    actions = np.empty(n_steps)
    temp = np.empty(n_steps)
    
    # Act on environment and save reward, hvac status and temperature
    for t in range(n_steps):
        action = {"0_1": agent.select_action(normStateDict(obs_dict["0_1"], config_dict))}
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        
        # Save data in arrays
        actions[t] = action["0_1"]
        reward[t] = rewards_dict["0_1"]
        hvac[t] = next_obs_dict["0_1"]["hvac_turned_on"]
        temp[t] = next_obs_dict["0_1"]["house_temp"]
        
        cumul_avg_reward += rewards_dict[k] / env.nb_agents
        
        obs_dict = next_obs_dict

    print(cumul_avg_reward/n_steps)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))
    ax1.plot(actions)
    ax1.plot(hvac)
    ax1.title.set_text('HVAC state vs. Agent action')
    ax2.plot(reward)
    ax2.title.set_text("Reward")
    ax3.plot(temp)
    ax3.title.set_text('Temperature')
    plt.show()

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from copy import deepcopy
from datetime import datetime, timedelta, time
from wandb_setup import wandb_setup


import wandb
import uuid
import os

def adjust_config(opt, config_dict):
    """Changes configuration of config_dict based on args."""
    if opt.nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["nb_agents"] = opt.nb_agents
    if opt.time_step != -1:
        config_dict["default_env_prop"]['time_step'] = opt.time_step
    if opt.cooling_capacity != -1:
        config_dict["default_hvac_prop"]['cooling_capacity'] = opt.cooling_capacity
    if opt.lockout_duration != -1:
        config_dict["default_hvac_prop"]['lockout_duration'] = opt.lockout_duration
    if opt.signal_mode != "config":
        config_dict["default_env_prop"]['power_grid_prop']["signal_mode"] = opt.signal_mode
    if opt.alpha != -1:
        config_dict["default_env_prop"]["alpha"] = opt.alpha

def render_and_wandb_init(opt, config_dict):
    render = opt.render

    log_wandb = not opt.no_wandb
    wandb_run = None
    if log_wandb:
        wandb_run = wandb_setup(opt, config_dict)
    return render, log_wandb, wandb_run


# Applying noise on environment properties
def applyPropertyNoise(default_env_prop, default_house_prop, noise_house_prop, default_hvac_prop, noise_hvac_prop):

    env_properties = deepcopy(default_env_prop)
    nb_agents = default_env_prop["cluster_prop"]["nb_agents"]

    # Creating the houses
    houses_properties = []
    agent_ids = []
    for i in range(nb_agents):
        house_prop = deepcopy(default_house_prop)
        apply_house_noise(house_prop, noise_house_prop)
        house_prop["id"] = str(i)
        hvac_prop = deepcopy(default_hvac_prop)
        apply_hvac_noise(hvac_prop, noise_hvac_prop)
        hvac_id = str(i) + "_1"
        hvac_prop["id"] = hvac_id
        agent_ids.append(hvac_id)
        house_prop["hvac_properties"] = [hvac_prop]
        houses_properties.append(house_prop)

    env_properties["cluster_prop"]["houses_properties"] = houses_properties
    env_properties["agent_ids"] = agent_ids
    env_properties["nb_hvac"] = len(agent_ids)

    # Setting the
    env_properties["start_datetime"] = get_random_date_time(datetime.strptime(
        default_env_prop["base_datetime"], "%Y-%m-%d %H:%M:%S"))  # Start date and time (Y,M,D, H, min, s)

    return env_properties


# Applying noise on properties
def apply_house_noise(house_prop, noise_house_prop):
    noise_house_mode = noise_house_prop["noise_mode"]
    noise_house_params = noise_house_prop["noise_parameters"][noise_house_mode]

    # Gaussian noise: target temp
    house_prop["init_temp"] += np.abs(random.gauss(0,
                                      noise_house_params["std_start_temp"]))
    house_prop["target_temp"] += np.abs(random.gauss(0,
                                        noise_house_params["std_target_temp"]))

    # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance

    factor_Ua = random.triangular(noise_house_params["factor_thermo_low"], noise_house_params["factor_thermo_high"], 1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ua"] *= factor_Ua

    factor_Cm = random.triangular(noise_house_params["factor_thermo_low"], noise_house_params["factor_thermo_high"], 1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Cm"] *= factor_Cm

    factor_Ca = random.triangular(noise_house_params["factor_thermo_low"], noise_house_params["factor_thermo_high"], 1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ca"] *= factor_Ca

    factor_Hm = random.triangular(noise_house_params["factor_thermo_low"], noise_house_params["factor_thermo_high"], 1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Hm"] *= factor_Hm


def apply_hvac_noise(hvac_prop, noise_hvac_prop):
    noise_hvac_mode = noise_hvac_prop["noise_mode"]
    noise_hvac_params = noise_hvac_prop["noise_parameters"][noise_hvac_mode]

    # Gaussian noise: latent_cooling_fraction
    hvac_prop["latent_cooling_fraction"] += random.gauss(
        0, noise_hvac_params["std_latent_cooling_fraction"])

    # Factor noise: COP, cooling_capacity
    factor_COP = random.triangular(noise_hvac_params["factor_COP_low"], noise_hvac_params["factor_COP_high"],
                                   1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    hvac_prop["COP"] *= factor_COP

    factor_cooling_capacity = random.triangular(noise_hvac_params["factor_cooling_capacity_low"],
                                                noise_hvac_params["factor_cooling_capacity_high"],
                                                1)  # low, high, mode ->  low <= N <= high, with max prob at mode.
    hvac_prop["cooling_capacity"] *= factor_cooling_capacity


def get_random_date_time(start_date_time):
    # Gets a uniformly sampled random date and time within a year from the start_date_time
    days_in_year = 364
    seconds_in_day = 60*60*24
    random_days = random.randrange(days_in_year)
    random_seconds = random.randrange(seconds_in_day)
    random_date = start_date_time + \
        timedelta(days=random_days, seconds=random_seconds)
    return random_date


# Multi agent management

def get_actions(actors, obs_dict):
    actions = {}
    for agent_id in actors.keys():
        actions[agent_id] = actors[agent_id].act(obs_dict)
    return actions


def datetime2List(dt):
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute]


def superDict2List(SDict, id):
    tmp = SDict[id].copy()
    tmp['datetime'] = datetime2List(tmp['datetime'])
    for k, v in tmp.items():
        if not isinstance(tmp[k], list):
            tmp[k] = [v]
    return sum(list(tmp.values()), [])


def normStateDict(sDict, config_dict, returnDict=False):
    default_house_prop = config_dict["default_house_prop"]
    default_hvac_prop = config_dict["default_hvac_prop"]
    default_env_prop = config_dict["default_env_prop"]

    result = {}
    k_temp = ['OD_temp', 'house_temp', 'house_mass_temp', 'house_target_temp']
    k_div = ['house_Ua', 'house_Cm', 'house_Ca', 'house_Hm', 'hvac_COP',
             'hvac_cooling_capacity', 'hvac_latent_cooling_fraction']
    # k_lockdown = ['hvac_seconds_since_off', 'hvac_lockout_duration']
    for k in k_temp:
        # Assuming the temperatures will be between 15 to 30, centered around 20 -> between -1 and 2, centered around 0.
        result[k] = (sDict[k]-20)/5
    result["house_deadband"] = sDict["house_deadband"]
    day = sDict['datetime'].timetuple().tm_yday
    hour = sDict['datetime'].hour
    result["sin_day"] = (np.sin(day*2*np.pi/365))
    result["cos_day"] = (np.cos(day*2*np.pi/365))
    result["sin_hr"] = np.sin(hour*2*np.pi/24)
    result["cos_hr"] = np.cos(hour*2*np.pi/24)
    for k in k_div:
        k1 = "_".join(k.split("_")[1:])
        if k1 in list(default_house_prop.keys()):
            result[k] = sDict[k]/default_house_prop[k1]
        elif k1 in list(default_hvac_prop.keys()):
            result[k] = sDict[k]/default_hvac_prop[k1]
        else:
            print(k)
            raise Exception("Error Key Matching.")
    result["hvac_turned_on"] = 1 if sDict["hvac_turned_on"] else 0
    result["hvac_lockout"] = 1 if sDict["hvac_lockout"] else 0

    result["hvac_seconds_since_off"] = sDict["hvac_seconds_since_off"] / \
        sDict["hvac_lockout_duration"]
    result["hvac_lockout_duration"] = sDict["hvac_lockout_duration"] / \
        sDict["hvac_lockout_duration"]

    result["reg_signal"] = sDict["reg_signal"] / \
        (default_env_prop["power_grid_prop"]["avg_power_per_hvac"]
         * default_env_prop["cluster_prop"]["nb_agents"])
    result["cluster_hvac_power"] = sDict["cluster_hvac_power"] / \
        (default_env_prop["power_grid_prop"]["avg_power_per_hvac"]
         * default_env_prop["cluster_prop"]["nb_agents"])
    return result if returnDict else np.array(list(result.values()))


def testAgentHouseTemperature(agent, state, low_temp, high_temp, config_dict, reg_signal):
    '''
    Receives an agent and a given state. Tests the agent probability output for 100 points a given range of indoors temperature, returning a vector for the probability of True (on).
    '''
    temp_range = np.linspace(low_temp, high_temp, num=100)
    prob_on = np.zeros(100)
    for i in range(100):
        temp = temp_range[i]
        state['house_temp'] = temp
        state['reg_signal'] = reg_signal
        norm_state = normStateDict(state, config_dict)

        action, action_prob = agent.select_action(norm_state)
        if not action:  # we want probability of True
            prob_on[i] = 1 - action_prob
        else:
            prob_on[i] = action_prob
    return prob_on


def colorPlotTestAgentHouseTemp(prob_on_per_training_on, prob_on_per_training_off, low_temp, high_temp, time_steps_test_log, log_wandb):
    '''
    Makes a color plot of the probability of the agent to turn on given indoors temperature, with the training
    '''

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,8.5), dpi=80)
    print(axes)

    normalizer = clr.Normalize(vmin=0, vmax=1)
    map0 = axes[0].imshow(np.transpose(prob_on_per_training_on), extent=[0, np.size(prob_on_per_training_on, 1)*time_steps_test_log, high_temp, low_temp], norm=normalizer)
    map1 = axes[1].imshow(np.transpose(prob_on_per_training_off), extent=[0, np.size(prob_on_per_training_off, 1)*time_steps_test_log, high_temp, low_temp], norm=normalizer)
    #axes[0] = plt.gca()
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    forceAspect(axes[0], aspect=2.0)
    forceAspect(axes[1], aspect=2.0)

    axes[0].set_xlabel("Training time steps")
    axes[1].set_xlabel("Training time steps")
    axes[0].set_ylabel("Indoors temperature")
    axes[1].set_ylabel("Indoors temperature")
    axes[0].set_title("Power: ON")
    axes[1].set_title("Power: OFF")

    cb = fig.colorbar(map0, ax=axes[:], shrink=0.6)

    if log_wandb:
        name = uuid.uuid1().hex + "probTestAgent.png"
        plt.savefig(name)
        wandb.log(
            {"Probability of agent vs Indoor temperature vs Episode ": wandb.Image(name)})
        os.remove(name)

    else:
        plt.show()
    return 0


def forceAspect(ax, aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def saveActorNetDict(agent, path):
    if not os.path.exists(path):
        os.makedirs(path)
    actor_net = agent.actor_net
    torch.save(actor_net.state_dict(), os.path.join(path, 'actor.pth'))

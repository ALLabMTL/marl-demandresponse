#%% Imports

import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

from copy import deepcopy
from datetime import datetime, timedelta, time

from wandb_setup import wandb_setup

#%% Functions


def render_and_wandb_init(opt, config_dict):
    render = opt.render
    log_wandb = not opt.no_wandb
    wandb_run = None
    if log_wandb:
        wandb_run = wandb_setup(opt, config_dict)
    return render, log_wandb, wandb_run


def adjust_config_train(opt, config_dict):
    """Changes configuration of config_dict based on args."""

### Environment
    if opt.nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["nb_agents"] = opt.nb_agents
    if opt.time_step != -1:
        config_dict["default_env_prop"]["time_step"] = opt.time_step

## Reward
    if opt.alpha_temp != -1:
        config_dict["default_env_prop"]["reward_prop"]["alpha_temp"] = opt.alpha_temp
    if opt.alpha_sig != -1:
        config_dict["default_env_prop"]["reward_prop"]["alpha_sig"] = opt.alpha_sig
    if opt.temp_penalty_mode != "config":
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_mode"] = opt.temp_penalty_mode
    if opt.alpha_ind_L2 != -1:
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_parameters"]["mixture"]["alpha_ind_L2"] = opt.alpha_ind_L2
    if opt.alpha_common_L2 != -1:
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_parameters"]["mixture"]["alpha_common_L2"] = opt.alpha_common_L2
    if opt.alpha_common_max != -1:
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_parameters"]["mixture"]["alpha_common_max"] = opt.alpha_common_max

## Simulator
# Outdoors
    if opt.OD_temp_mode != "config":
        config_dict["default_env_prop"]["cluster_prop"]["temp_mode"] = opt.OD_temp_mode
    config_dict["default_house_prop"]["solar_gain_bool"] = not opt.no_solar_gain
# House and HVAC
    if opt.cooling_capacity != -1:
        config_dict["default_hvac_prop"]["cooling_capacity"] = opt.cooling_capacity
    if opt.lockout_duration != -1:
        config_dict["default_hvac_prop"]["lockout_duration"] = opt.lockout_duration
# Noise
    if opt.house_noise_mode != "config":
        config_dict["noise_house_prop"]["noise_mode"] = opt.house_noise_mode
    if opt.house_noise_mode_test == "train":
        config_dict["noise_house_prop_test"]["noise_mode"] = config_dict["noise_house_prop"]["noise_mode"]
    else:
        config_dict["noise_house_prop_test"]["noise_mode"] = opt.house_noise_mode_test
    if opt.hvac_noise_mode != "config":
        config_dict["noise_hvac_prop"]["noise_mode"] = opt.hvac_noise_mode
    if opt.hvac_noise_mode_test == "train":
        config_dict["noise_hvac_prop_test"]["noise_mode"] = config_dict["noise_hvac_prop_test"]["noise_mode"]
    else:
        config_dict["noise_hvac_prop_test"]["noise_mode"] = opt.hvac_noise_mode_test

## Signal
    if opt.signal_mode != "config":
        config_dict["default_env_prop"]["power_grid_prop"]["signal_mode"] = opt.signal_mode
    if opt.base_power_mode != "config":
        config_dict["default_env_prop"]["power_grid_prop"]["base_power_mode"] = opt.base_power_mode
    config_dict["default_env_prop"]["power_grid_prop"]["artificial_ratio"] = opt.artificial_signal_ratio
    if opt.artificial_signal_ratio_range != -1:
        config_dict["default_env_prop"]["power_grid_prop"]["artificial_signal_ratio_range"] = opt.artificial_signal_ratio_range


## State
    config_dict["default_env_prop"]["state_properties"]["solar_gain"] = opt.state_solar_gain == 'True'
    config_dict["default_env_prop"]["state_properties"]["hour"] = opt.state_hour == 'True'
    config_dict["default_env_prop"]["state_properties"]["day"] = opt.state_day == 'True'
    config_dict["default_env_prop"]["state_properties"]["thermal"] = opt.state_thermal == 'True'


### Agent

## Agent communication constraints

    if opt.nb_agents_comm != -1:
        config_dict["default_env_prop"]["cluster_prop"]["nb_agents_comm"] = opt.nb_agents_comm
    if opt.agents_comm_mode != "config":
        config_dict["default_env_prop"]["cluster_prop"]["agents_comm_mode"] = opt.agents_comm_mode

## PPO agent
# NN architecture
    if opt.layers_actor != "config":
        config_dict["PPO_prop"]["actor_layers"] = opt.layers_actor
    if opt.layers_critic != "config":
        config_dict["PPO_prop"]["critic_layers"] = opt.layers_critic
    if opt.layers_both != "config":
        config_dict["PPO_prop"]["actor_layers"] = opt.layers_both
        config_dict["PPO_prop"]["critic_layers"] = opt.layers_both
# NN optimization
    if opt.batch_size != -1:
        config_dict["PPO_prop"]["batch_size"] = opt.batch_size
    if opt.lr_critic != -1:
        config_dict["PPO_prop"]["lr_critic"] = opt.lr_critic
    if opt.lr_actor != -1:
        config_dict["PPO_prop"]["lr_actor"] = opt.lr_actor
    if opt.lr_both != -1:
        config_dict["PPO_prop"]["lr_critic"] = opt.lr_both
        config_dict["PPO_prop"]["lr_actor"] = opt.lr_both
        if opt.lr_actor != -1 or opt.lr_critic != -1:
            raise ValueError("Potential conflict: both lr_both and lr_actor or lr_critic were set in the CLI")
# RL optimization
    if opt.gamma != -1:
        config_dict["PPO_prop"]["gamma"] = opt.gamma
    if opt.clip_param != -1:
        config_dict["PPO_prop"]["clip_param"] = opt.clip_param
    if opt.max_grad_norm != -1:
        config_dict["PPO_prop"]["max_grad_norm"] = opt.max_grad_norm
    if opt.ppo_update_time != -1:
        config_dict["PPO_prop"]["ppo_update_time"] = opt.ppo_update_time

## DQN agent
# NN architecture
    if opt.DQNnetwork_layers != "config":
        config_dict["DQN_prop"]["network_layers"] = opt.DQNnetwork_layers

# NN optimization
    if opt.batch_size != -1:
        config_dict["DQN_prop"]["batch_size"] = opt.batch_size
    if opt.DQN_lr != -1:
        config_dict["DQN_prop"]["lr"] = opt.DQN_lr       

# RL optimization
    if opt.gamma != -1:
        config_dict["DQN_prop"]["gamma"] = opt.gamma
    if opt.tau != -1:
        config_dict["DQN_prop"]["tau"] = opt.tau
    if opt.buffer_capacity != -1:
        config_dict["DQN_prop"]["buffer_capacity"] = opt.buffer_capacity    
    if opt.epsilon_decay != -1:
        config_dict["DQN_prop"]["epsilon_decay"] = opt.epsilon_decay    
    if opt.min_epsilon != -1:
        config_dict["DQN_prop"]["min_epsilon"] = opt.min_epsilon    
        
def adjust_config_deploy(opt, config_dict):
    if opt.nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["nb_agents"] = opt.nb_agents
    if opt.time_step != -1:
        config_dict["default_env_prop"]["time_step"] = opt.time_step
    if opt.cooling_capacity != -1:
        config_dict["default_hvac_prop"]["cooling_capacity"] = opt.cooling_capacity
    if opt.lockout_duration != -1:
        config_dict["default_hvac_prop"]["lockout_duration"] = opt.lockout_duration
    if opt.MPC_rolling_horizon != -1:
        config_dict["MPC_prop"]["rolling_horizon"] = opt.MPC_rolling_horizon
    if opt.signal_mode != "config":
        config_dict["default_env_prop"]["power_grid_prop"][
            "signal_mode"
        ] = opt.signal_mode
    if opt.house_noise_mode != "config":
        config_dict["noise_house_prop"]["noise_mode"] = opt.house_noise_mode
    if opt.hvac_noise_mode != "config":
        config_dict["noise_hvac_prop"]["noise_mode"] = opt.hvac_noise_mode
    if opt.OD_temp_mode != "config":
        config_dict["default_env_prop"]["cluster_prop"]["temp_mode"] = opt.OD_temp_mode
    config_dict["default_house_prop"]["solar_gain_bool"] = not opt.no_solar_gain
    if opt.base_power_mode != "config":
        config_dict["default_env_prop"]["power_grid_prop"][
            "base_power_mode"
        ] = opt.base_power_mode
    if opt.nb_agents_comm != -1:
        config_dict["default_env_prop"]["cluster_prop"][
            "nb_agents_comm"
        ] = opt.nb_agents_comm
    if opt.agents_comm_mode != "config":
        config_dict["default_env_prop"]["cluster_prop"][
            "agents_comm_mode"
        ] = opt.agents_comm_mode
    if opt.layers_actor != "config":
        config_dict["PPO_prop"]["actor_layers"] = opt.layers_actor
    if opt.layers_critic != "config":
        config_dict["PPO_prop"]["critic_layers"] = opt.layers_critic
    if opt.layers_both != "config":
        config_dict["PPO_prop"]["actor_layers"] = opt.layers_both
        config_dict["PPO_prop"]["critic_layers"] = opt.layers_both
    if opt.DQNnetwork_layers != "config":
        config_dict["DQN_prop"]["network_layers"] = opt.DQNnetwork_layers
    if opt.start_datetime_mode != "config":
        config_dict["default_env_prop"]["start_datetime_mode"] = opt.start_datetime_mode

    config_dict["default_env_prop"]["state_properties"]["solar_gain"] = opt.state_solar_gain == 'True'
    config_dict["default_env_prop"]["state_properties"]["hour"] = opt.state_hour == 'True'
    config_dict["default_env_prop"]["state_properties"]["day"] = opt.state_day == 'True'
    config_dict["default_env_prop"]["state_properties"]["thermal"] = opt.state_thermal == 'True'

    config_dict["default_env_prop"]["power_grid_prop"]["artificial_ratio"] = opt.artificial_signal_ratio


# Applying noise on environment properties
def applyPropertyNoise(
    default_env_prop,
    default_house_prop,
    noise_house_prop,
    default_hvac_prop,
    noise_hvac_prop,
):

    env_properties = deepcopy(default_env_prop)
    nb_agents = default_env_prop["cluster_prop"]["nb_agents"]

    # Creating the houses
    houses_properties = []
    agent_ids = []
    for i in range(nb_agents):
        house_prop = deepcopy(default_house_prop)
        apply_house_noise(house_prop, noise_house_prop)
        house_id = i
        house_prop["id"] = house_id
        hvac_prop = deepcopy(default_hvac_prop)
        apply_hvac_noise(hvac_prop, noise_hvac_prop)
        hvac_prop["id"] = house_id
        agent_ids.append(house_id)
        house_prop["hvac_properties"] = hvac_prop
        houses_properties.append(house_prop)

    env_properties["cluster_prop"]["houses_properties"] = houses_properties
    env_properties["agent_ids"] = agent_ids
    env_properties["nb_hvac"] = len(agent_ids)

    # Setting the date
    if env_properties["start_datetime_mode"] == "random":
        env_properties["start_datetime"] = get_random_date_time(
            datetime.strptime(default_env_prop["start_datetime"], "%Y-%m-%d %H:%M:%S")
        )  # Start date and time (Y,M,D, H, min, s)
    elif env_properties["start_datetime_mode"] == "fixed":
        env_properties["start_datetime"] = datetime.strptime(
            default_env_prop["start_datetime"], "%Y-%m-%d %H:%M:%S"
        )
    else:
        raise ValueError(
            "start_datetime_mode in default_env_prop in config.py must be random or fixed. Current value: {}.".format(
                env_properties["start_datetime_mode"] == "fixed"
            )
        )

    return env_properties


# Applying noise on properties
def apply_house_noise(house_prop, noise_house_prop):
    noise_house_mode = noise_house_prop["noise_mode"]
    noise_house_params = noise_house_prop["noise_parameters"][noise_house_mode]

    # Gaussian noise: target temp
    house_prop["init_air_temp"] += np.abs(
        random.gauss(0, noise_house_params["std_start_temp"])
    )
    house_prop["init_mass_temp"] += np.abs(
        random.gauss(0, noise_house_params["std_start_temp"])
    )
    house_prop["target_temp"] += np.abs(
        random.gauss(0, noise_house_params["std_target_temp"])
    )

    # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance

    factor_Ua = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ua"] *= factor_Ua

    factor_Cm = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Cm"] *= factor_Cm

    factor_Ca = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ca"] *= factor_Ca

    factor_Hm = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Hm"] *= factor_Hm


def apply_hvac_noise(hvac_prop, noise_hvac_prop):
    noise_hvac_mode = noise_hvac_prop["noise_mode"]
    hvac_capacity = hvac_prop["cooling_capacity"]
    noise_hvac_params = noise_hvac_prop["noise_parameters"][noise_hvac_mode]

    hvac_prop["cooling_capacity"] = random.choices(noise_hvac_params["cooling_capacity_list"][hvac_capacity])[0]


"""
    # Gaussian noise: latent_cooling_fraction
    hvac_prop["latent_cooling_fraction"] += random.gauss(
        0, noise_hvac_params["std_latent_cooling_fraction"]
    )

    # Factor noise: COP, cooling_capacity
    factor_COP = random.triangular(
        noise_hvac_params["factor_COP_low"], noise_hvac_params["factor_COP_high"], 1
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.

    hvac_prop["COP"] *= factor_COP

    factor_cooling_capacity = random.triangular(
        noise_hvac_params["factor_cooling_capacity_low"],
        noise_hvac_params["factor_cooling_capacity_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    hvac_prop["cooling_capacity"] *= factor_cooling_capacity
"""


def get_random_date_time(start_date_time):
    # Gets a uniformly sampled random date and time within a year from the start_date_time
    days_in_year = 364
    seconds_in_day = 60 * 60 * 24
    random_days = random.randrange(days_in_year)
    random_seconds = random.randrange(seconds_in_day)
    random_date = start_date_time + timedelta(days=random_days, seconds=random_seconds)

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
    tmp["datetime"] = datetime2List(tmp["datetime"])
    for k, v in tmp.items():
        if not isinstance(tmp[k], list):
            tmp[k] = [v]
    return sum(list(tmp.values()), [])


def normStateDict(sDict, config_dict, returnDict=False):
    default_house_prop = config_dict["default_house_prop"]
    default_hvac_prop = config_dict["default_hvac_prop"]
    default_env_prop = config_dict["default_env_prop"]
    state_prop = default_env_prop["state_properties"]

    result = {}
    if state_prop["thermal"]:
        k_temp = ["OD_temp", "house_temp", "house_mass_temp", "house_target_temp"]
        k_div = [
            "house_Ua",
            "house_Cm",
            "house_Ca",
            "house_Hm",
            "hvac_COP",
            "hvac_cooling_capacity",
            "hvac_latent_cooling_fraction",
        ]
    else:
        k_temp = ["house_temp", "house_mass_temp", "house_target_temp"]    
        k_div = ["hvac_cooling_capacity"]    

    # k_lockdown = ['hvac_seconds_since_off', 'hvac_lockout_duration']
    for k in k_temp:
        # Assuming the temperatures will be between 15 to 30, centered around 20 -> between -1 and 2, centered around 0.
        result[k] = (sDict[k] - 20) / 5
    result["house_deadband"] = sDict["house_deadband"]

    if state_prop["day"]:
        day = sDict["datetime"].timetuple().tm_yday
        result["sin_day"] = np.sin(day * 2 * np.pi / 365)
        result["cos_day"] = np.cos(day * 2 * np.pi / 365)
    if state_prop["hour"]:
        hour = sDict["datetime"].hour
        result["sin_hr"] = np.sin(hour * 2 * np.pi / 24)
        result["cos_hr"] = np.cos(hour * 2 * np.pi / 24)

    if state_prop["solar_gain"]:
        result["house_solar_gain"] = sDict["house_solar_gain"]/1000

    for k in k_div:
        k1 = "_".join(k.split("_")[1:])
        if k1 in list(default_house_prop.keys()):
            result[k] = sDict[k] / default_house_prop[k1]
        elif k1 in list(default_hvac_prop.keys()):
            result[k] = sDict[k] / default_hvac_prop[k1]
        else:
            print(k)
            raise Exception("Error Key Matching.")
    result["hvac_turned_on"] = 1 if sDict["hvac_turned_on"] else 0
    result["hvac_lockout"] = 1 if sDict["hvac_lockout"] else 0

    result["hvac_seconds_since_off"] = (
        sDict["hvac_seconds_since_off"] / sDict["hvac_lockout_duration"]
    )
    result["hvac_lockout_duration"] = (
        sDict["hvac_lockout_duration"] / sDict["hvac_lockout_duration"]
    )

    result["reg_signal"] = sDict["reg_signal"] / (
        default_env_prop["reward_prop"]["norm_reg_sig"]
        * default_env_prop["cluster_prop"]["nb_agents"]
    )
    result["cluster_hvac_power"] = sDict["cluster_hvac_power"] / (
        default_env_prop["reward_prop"]["norm_reg_sig"]
        * default_env_prop["cluster_prop"]["nb_agents"]
    )

    temp_messages = []
    for message in sDict["message"]:
        r_message = {}
        r_message["current_temp_diff_to_target"] = (
            message["current_temp_diff_to_target"] / 5
        )  # Already a difference, only need to normalize like k_temps
        r_message["hvac_seconds_since_off"] = (
            message["hvac_seconds_since_off"] / sDict["hvac_lockout_duration"]
        )
        r_message["hvac_curr_consumption"] = (
            message["hvac_curr_consumption"]
            / default_env_prop["reward_prop"]["norm_reg_sig"]
        )
        r_message["hvac_max_consumption"] = (
            message["hvac_max_consumption"]
            / default_env_prop["reward_prop"]["norm_reg_sig"]
        )
        temp_messages.append(r_message)

    if returnDict:
        result["message"] = temp_messages

    else:  # Flatten the dictionary in a single np_array
        flat_messages = []
        for message in temp_messages:
            flat_message = list(message.values())
            flat_messages = flat_messages + flat_message
        result = np.array(list(result.values()) + flat_messages)

    return result


#%% Testing


def test_dqn_agent(agent, env, config_dict, opt, tr_time_steps):
    """
    Test dqn agent on an episode of nb_test_timesteps
    """
    env = deepcopy(env)
    cumul_avg_reward = 0
    cumul_temp_error = 0
    cumul_signal_error = 0

    obs_dict = env.reset()
    with torch.no_grad():
        for t in range(opt.nb_time_steps_test):
            action = {
                k: agent.select_action(normStateDict(obs_dict[k], config_dict))
                for k in obs_dict.keys()
            }
            obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            for i in range(env.nb_agents):
                cumul_avg_reward += rewards_dict[i] / env.nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.nb_agents
                )
                cumul_signal_error += np.abs(
                    obs_dict[i]["reg_signal"] - obs_dict[i]["cluster_hvac_power"]
                ) / (env.nb_agents**2)

    mean_avg_return = cumul_avg_reward / opt.nb_time_steps_test
    mean_temp_error = cumul_temp_error / opt.nb_time_steps_test
    mean_signal_error = cumul_signal_error / opt.nb_time_steps_test
    
    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean signal error": mean_signal_error,
        "Training steps": tr_time_steps,
    }


def test_ppo_agent(agent, env, config_dict, opt, tr_time_steps):
    """
    Test ppo agent on an episode of nb_test_timesteps, with
    """
    env = deepcopy(env)
    cumul_avg_reward = 0
    cumul_temp_error = 0
    cumul_signal_error = 0
    obs_dict = env.reset()
    with torch.no_grad():
        for t in range(opt.nb_time_steps_test):
            action_and_prob = {
                k: agent.select_action(normStateDict(obs_dict[k], config_dict))
                for k in obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
            obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            for i in range(env.nb_agents):
                cumul_avg_reward += rewards_dict[i] / env.nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.nb_agents
                )
                cumul_signal_error += np.abs(
                    obs_dict[i]["reg_signal"] - obs_dict[i]["cluster_hvac_power"]
                ) / (env.nb_agents**2)
    mean_avg_return = cumul_avg_reward / opt.nb_time_steps_test
    mean_temp_error = cumul_temp_error / opt.nb_time_steps_test
    mean_signal_error = cumul_signal_error / opt.nb_time_steps_test

    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean signal error": mean_signal_error,
        "Training steps": tr_time_steps,
    }



def saveActorNetDict(agent, path, t=None):
    if not os.path.exists(path):
        os.makedirs(path)
    actor_net = agent.actor_net
    if t:
        torch.save(actor_net.state_dict(), os.path.join(path, "actor" + str(t) + ".pth"))
    else:
        torch.save(actor_net.state_dict(), os.path.join(path, "actor.pth"))

def saveDQNNetDict(agent, path, t=None):
    if not os.path.exists(path):
        os.makedirs(path)
    policy_net = agent.policy_net
    if t:
        torch.save(policy_net.state_dict(), os.path.join(path, "DQN" + str(t) + ".pth"))
    else:
        torch.save(policy_net.state_dict(), os.path.join(path, "DQN.pth"))


def clipInterpolationPoint(point, parameter_dict):
    for key in point.keys():
        values = np.array(parameter_dict[key])
        if point[key] > np.max(values):
            point[key] = np.max(values)
        elif point[key] < np.min(values):
            point[key] = np.min(values)
    return point


def sortDictKeys(point, dict_keys):
    point2 = {}
    for key in dict_keys:
        point2[key] = point[key]
    return point2


class Perlin:
    def __init__(self, amplitude, nb_octaves, octaves_step, period, seed):

        self.amplitude = amplitude
        self.nb_octaves = nb_octaves
        self.octaves_step = octaves_step
        self.period = period

        self.seed = seed

        self.noise_list = []
        for i in range(self.nb_octaves):
            self.noise_list.append(
                PerlinNoise(octaves=2**i * octaves_step, seed=seed)
            )

    def calculate_noise(self, x):
        noise = 0

        for j in range(self.nb_octaves - 1):
            noise += self.noise_list[j].noise(x / self.period) / (2**j)
        noise += self.noise_list[-1].noise(x / self.period) / (2**self.nb_octaves - 1)
        return self.amplitude * noise

    def plot_noise(self, timesteps=500):
        l = []

        for x in range(timesteps):
            noise = self.calculate_noise(x)
            l.append(noise)

        plt.plot(l)
        plt.show()


def deadbandL2(target, deadband, value):
    if target + deadband / 2 < value:
        deadband_L2 = (value - (target + deadband / 2)) ** 2
    elif target - deadband / 2 > value:
        deadband_L2 = ((target - deadband / 2) - value) ** 2
    else:
        deadband_L2 = 0.0

    return deadband_L2


def house_solar_gain(date_time, window_area, shading_coeff):
    """
    Computes the solar gain, i.e. the heat transfer received from the sun through the windows.

    Return:
    solar_gain: float, direct solar radiation passing through the windows at a given moment in Watts

    Par¸ameters
    date_time: datetime, current date and time

    ---
    Source and assumptions:
    CIBSE. (2015). Environmental Design - CIBSE Guide A (8th Edition) - 5.9.7 Solar Cooling Load Tables. CIBSE.
    Retrieved from https://app.knovel.com/hotlink/pdf/id:kt0114THK9/environmental-design/solar-cooling-load-tables
    Table available: https://www.cibse.org/Knowledge/Guide-A-2015-Supplementary-Files/Chapter-5

    Coefficient obtained by performing a polynomial regression on the table "solar cooling load at stated sun time at latitude 30".

    Based on the following assumptions.
    - Latitude is 30. (The latitude of Austin in Texas is 30.266666)
    - The SCL before 7:30 and after 17:30 is negligible for latitude 30.
    - The windows are distributed perfectly evenly around the building.
    - There are no horizontal windows, for example on the roof.
    """

    x = date_time.hour + date_time.minute / 60 - 7.5
    if x < 0 or x > 10:
        solar_cooling_load = 0
    else:
        y = date_time.month + date_time.day / 30 - 1
        coeff = [
            4.36579418e01,
            1.58055357e02,
            8.76635241e01,
            -4.55944821e01,
            3.24275366e00,
            -4.56096472e-01,
            -1.47795612e01,
            4.68950855e00,
            -3.73313090e01,
            5.78827663e00,
            1.04354810e00,
            2.12969604e-02,
            2.58881400e-03,
            -5.11397219e-04,
            1.56398008e-02,
            -1.18302764e-01,
            -2.71446436e-01,
            -3.97855577e-02,
        ]

        solar_cooling_load = (
            coeff[0]
            + x * coeff[1]
            + y * coeff[2]
            + x**2 * coeff[3]
            + x**2 * y * coeff[4]
            + x**2 * y**2 * coeff[5]
            + y**2 * coeff[6]
            + x * y**2 * coeff[7]
            + x * y * coeff[8]
            + x**3 * coeff[9]
            + y**3 * coeff[10]
            + x**3 * y * coeff[11]
            + x**3 * y**2 * coeff[12]
            + x**3 * y**3 * coeff[13]
            + x**2 * y**3 * coeff[14]
            + x * y**3 * coeff[15]
            + x**4 * coeff[16]
            + y**4 * coeff[17]
        )

    solar_gain = window_area * shading_coeff * solar_cooling_load
    return solar_gain

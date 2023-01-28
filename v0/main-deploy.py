# from apt import ProblemResolver
import os
import random
import time
from cmath import nan

import numpy as np
import pandas as pd

from .agents import (
    AlwaysOnController,
    BangBangController,
    BasicController,
    DDPGAgent,
    DeadbandBangBangController,
    DQNAgent,
    GreedyMyopic,
    MPCController,
    PPOAgent,
)
from .cli import cli_deploy
from .config import config_dict
from .env.MA_DemandResponse import MADemandResponseEnv
from .utils import adjust_config_deploy, get_actions, normStateDict
from .wandb_setup import wandb_setup

os.environ["WANDB_SILENT"] = "true"

agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "PPO": PPOAgent,
    "MAPPO": PPOAgent,
    "DQN": DQNAgent,
    "GreedyMyopic": GreedyMyopic,
    "MPC": MPCController,
    "MADDPG": DDPGAgent,
}


# CLI arguments

opt = cli_deploy(agents_dict)
adjust_config_deploy(opt, config_dict)

log_wandb = not opt.no_wandb
if opt.render:
    from env.renderer import Renderer

    renderer = Renderer(opt.nb_agents)
if opt.log_metrics_path != "":
    df_metrics = pd.DataFrame()

# Creating environment
random.seed(opt.env_seed)
nb_time_steps = opt.nb_time_steps

if log_wandb:
    wandb_run = wandb_setup(opt, config_dict)

env = MADemandResponseEnv(config_dict)
obs_dict = env.reset()

num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))

if opt.log_metrics_path != "":
    df_metrics = pd.DataFrame()
time_steps_log = int(opt.nb_time_steps / opt.nb_logs)
nb_agents = config_dict["default_env_prop"]["cluster_prop"]["nb_agents"]
houses = env.cluster.houses

actors = {}
for house_id in houses.keys():
    agent_prop = {"id": house_id}

    if opt.actor_name:
        agent_prop["actor_name"] = opt.actor_name
        agent_prop["net_seed"] = opt.net_seed

    actors[house_id] = agents_dict[opt.agent](
        agent_prop, config_dict, num_state=num_state
    )


obs_dict = env.reset()


cumul_temp_offset = 0
cumul_temp_error = 0
max_temp_error = 0
cumul_signal_offset = 0
cumul_signal_error = 0
cumul_OD_temp = 0
cumul_signal = 0
cumul_cons = 0

cumul_squared_error_sig = 0
cumul_squared_error_temp = 0
cumul_squared_max_error_temp = 0

actions = get_actions(actors, obs_dict)
t1_start = time.process_time()

for i in range(nb_time_steps):
    obs_dict, _, _, info = env.step(actions)
    actions = get_actions(actors, obs_dict)
    if opt.log_metrics_path != "" and i >= opt.start_stats_from:
        df = pd.DataFrame(obs_dict).transpose()

        df["temperature_difference"] = df["house_temp"] - df["house_target_temp"]
        df["temperature_error"] = np.abs(df["house_temp"] - df["house_target_temp"])
        temp_diff = df["temperature_difference"].mean()
        temp_err = df["temperature_error"].mean()
        air_temp = df["house_temp"].mean()
        mass_temp = df["house_mass_temp"].mean()
        target_temp = df["house_target_temp"].mean()
        OD_temp = df["OD_temp"][0]
        signal = df["reg_signal"][0]
        consumption = df["cluster_hvac_power"][0]
        row = pd.DataFrame(
            {
                "temp_diff": temp_diff,
                "temp_err": temp_err,
                "air_temp": air_temp,
                "mass_temp": mass_temp,
                "target_temp": target_temp,
                "OD_temp": OD_temp,
                "signal": signal,
                "consumption": consumption,
            },
            index=[config_dict["default_env_prop"]["time_step"] * i],
        )
        df_metrics = pd.concat([df_metrics, row])

    if opt.render and i >= opt.render_after:
        renderer.render(obs_dict)
    max_temp_error_houses = 0
    for k in obs_dict.keys():
        temp_error = obs_dict[k]["house_temp"] - obs_dict[k]["house_target_temp"]
        cumul_temp_offset += temp_error / env.nb_agents
        cumul_temp_error += np.abs(temp_error) / env.nb_agents
        if np.abs(temp_error) > max_temp_error:
            max_temp_error = np.abs(temp_error)
        if np.abs(temp_error) > max_temp_error_houses:
            max_temp_error_houses = np.abs(temp_error)

        if i >= opt.start_stats_from:
            cumul_squared_error_temp += temp_error**2

    if i >= opt.start_stats_from:
        cumul_squared_max_error_temp += max_temp_error_houses**2
    cumul_OD_temp += obs_dict[0]["OD_temp"]
    cumul_signal += obs_dict[0]["reg_signal"]
    cumul_cons += obs_dict[0]["cluster_hvac_power"]

    signal_error = obs_dict[0]["reg_signal"] - obs_dict[0]["cluster_hvac_power"]
    cumul_signal_offset += signal_error
    cumul_signal_error += np.abs(signal_error)

    if i >= opt.start_stats_from:
        cumul_squared_error_sig += signal_error**2

    if i % time_steps_log == time_steps_log - 1:  # Log train statistics
        # print("Logging stats at time {}".format(t))

        # print("Average absolute noise: {} W".format(env.power_grid.cumulated_abs_noise / env.power_grid.nb_steps ))

        mean_temp_offset = cumul_temp_offset / time_steps_log
        mean_temp_error = cumul_temp_error / time_steps_log
        mean_signal_offset = cumul_signal_offset / time_steps_log
        mean_signal_error = cumul_signal_error / time_steps_log
        mean_OD_temp = cumul_OD_temp / time_steps_log
        mean_signal = cumul_signal / time_steps_log
        mean_consumption = cumul_cons / time_steps_log

        if i >= opt.start_stats_from:
            rmse_sig_per_ag = (
                np.sqrt(cumul_squared_error_sig / (i - opt.start_stats_from))
                / env.nb_agents
            )
            rmse_temp = np.sqrt(
                cumul_squared_error_temp / ((i - opt.start_stats_from) * env.nb_agents)
            )
            rms_max_error_temp = np.sqrt(
                cumul_squared_max_error_temp / (i - opt.start_stats_from)
            )
        else:
            rmse_sig_per_ag = nan
            rmse_temp = nan
            rms_max_error_temp = nan

        if log_wandb:
            wandb_run.log(
                {
                    "RMSE signal per agent": rmse_sig_per_ag,
                    "RMSE temperature": rmse_temp,
                    "RMS Max Error temperature": rms_max_error_temp,
                    "Mean temperature offset": mean_temp_offset,
                    "Mean temperature error": mean_temp_error,
                    "Max temperature error": max_temp_error,
                    "Mean signal offset": mean_signal_offset,
                    "Mean signal error": mean_signal_error,
                    "Mean outside temperature": mean_OD_temp,
                    "Mean signal": mean_signal,
                    "Mean consumption": mean_consumption,
                    "Time (hour)": obs_dict[0]["datetime"].hour,
                    "Time step": i,
                }
            )

        cumul_temp_offset = 0
        cumul_temp_error = 0
        max_temp_error = 0
        cumul_signal_offset = 0
        cumul_signal_error = 0
        cumul_OD_temp = 0
        cumul_signal = 0
        cumul_cons = 0
        print("Time step: {}".format(i))
        t1_stop = time.process_time()
        print(
            "Elapsed time for {}% of steps: {} seconds.".format(
                int(np.round(float(i) / nb_time_steps * 100)), int(t1_stop - t1_start)
            )
        )

rmse_sig_per_ag = (
    np.sqrt(cumul_squared_error_sig / (nb_time_steps - opt.start_stats_from))
    / env.nb_agents
)
rmse_temp = np.sqrt(
    cumul_squared_error_temp / ((nb_time_steps - opt.start_stats_from) * env.nb_agents)
)
rms_max_error_temp = np.sqrt(
    cumul_squared_max_error_temp / (nb_time_steps - opt.start_stats_from)
)
print("RMSE Signal per agent: {} W".format(int(rmse_sig_per_ag)))
print("RMSE Temperature: {} C".format(rmse_temp))
print("RMS Max Error Temperature: {} C".format(rms_max_error_temp))


# print("Average absolute noise: {} W".format(env.power_grid.cumulated_abs_noise / env.power_grid.nb_steps ))
if log_wandb:
    wandb_run.log(
        {
            "RMSE signal per agent": rmse_sig_per_ag,
            "RMSE temperature": rmse_temp,
            "RMS Max Error temperature": rms_max_error_temp,
        }
    )
if opt.log_metrics_path != "":
    df_metrics.to_csv(opt.log_metrics_path)

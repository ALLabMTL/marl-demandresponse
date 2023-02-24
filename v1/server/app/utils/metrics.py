import numpy as np

from app.core.environment.environment import Environment


class Metrics:
    def __init__(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        self.cumul_next_signal_offset = 0
        self.cumul_next_signal_error = 0

    def update(self, k, obs_dict, next_obs_dict, rewards_dict, env: Environment):
        nb_agents = len(env.cluster.buildings)
        self.cumul_temp_offset += (
            next_obs_dict[k]["indoor_temp"] - next_obs_dict[k]["target_temp"]
        ) / len(env.cluster.buildings)
        self.cumul_temp_error += (
            np.abs(next_obs_dict[k]["indoor_temp"] - next_obs_dict[k]["target_temp"])
            / nb_agents
        )
        self.cumul_avg_reward += rewards_dict[k] / nb_agents
        self.cumul_next_signal_offset += (
            next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"]
        ) / (nb_agents**2)
        self.cumul_next_signal_error += np.abs(
            next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"]
        ) / (nb_agents**2)
        self.cumul_signal_offset += (
            obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"]
        ) / (nb_agents**2)
        self.cumul_signal_error += np.abs(
            obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"]
        ) / (nb_agents**2)

    def log(self, t, time_steps_train_log):
        mean_avg_return = self.cumul_avg_reward / time_steps_train_log
        mean_temp_offset = self.cumul_temp_offset / time_steps_train_log
        mean_temp_error = self.cumul_temp_error / time_steps_train_log
        mean_next_signal_offset = self.cumul_next_signal_offset / time_steps_train_log
        mean_next_signal_error = self.cumul_next_signal_error / time_steps_train_log
        mean_signal_offset = self.cumul_signal_offset / time_steps_train_log
        mean_signal_error = self.cumul_signal_error / time_steps_train_log
        metrics = {
            "Mean train return": mean_avg_return,
            "Mean temperature offset": mean_temp_offset,
            "Mean temperature error": mean_temp_error,
            "Mean next signal offset": mean_next_signal_offset,
            "Mean next signal error": mean_next_signal_error,
            "Mean signal error": mean_signal_error,
            "Mean signal offset": mean_signal_offset,
            "Training steps": t,
        }
        return metrics

    def reset(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        self.cumul_next_signal_offset = 0
        self.cumul_next_signal_error = 0

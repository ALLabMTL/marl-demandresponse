import numpy as np

class Metrics:
    def __init__(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        
    def update(self, k, next_obs_dict, rewards_dict, env):
        self.cumul_temp_offset += (next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / env.nb_agents
        self.cumul_temp_error += np.abs(next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / env.nb_agents
        self.cumul_avg_reward += rewards_dict[k] / env.nb_agents
        self.cumul_signal_offset += next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"]
        self.cumul_signal_error += np.abs(next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"])
    
    def log(self, t, time_steps_train_log):
        mean_avg_return = self.cumul_avg_reward / time_steps_train_log
        mean_temp_offset = self.cumul_temp_offset / time_steps_train_log
        mean_temp_error = self.cumul_temp_error / time_steps_train_log
        mean_signal_offset = self.cumul_signal_offset / time_steps_train_log
        mean_signal_error = self.cumul_signal_error / time_steps_train_log
        metrics = {"Mean train return": mean_avg_return,
                   "Mean temperature offset": mean_temp_offset,
                   "Mean temperature error": mean_temp_error,
                   "Mean signal offset": mean_signal_offset,
                   "Mean signal error": mean_signal_error,
                   "Training steps": t}
        return metrics
    
    def reset(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
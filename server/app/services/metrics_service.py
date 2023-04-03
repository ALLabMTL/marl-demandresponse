from cmath import nan
from datetime import datetime

import numpy as np

from app.services.wandb_service import WandbManager
from app.utils.logger import logger


class Metrics:
    def __init__(
        self,
        wandb_service: WandbManager,
    ) -> None:
        self.wandb_service = wandb_service

    def initialize(
        self, nb_agents: int, start_stats_from: int, nb_time_steps: int
    ) -> None:
        self.nb_time_steps = nb_time_steps
        self.start_stats_from = start_stats_from
        self.nb_agents = nb_agents
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        self.cumul_squared_error_temp = 0
        self.max_temp_error = 0
        self.cumul_OD_temp = 0
        self.cumul_signal = 0
        self.cumul_cons = 0
        self.cumul_squared_error_sig = 0
        self.rmse_sig_per_ag = nan
        self.rmse_temp = nan
        self.rms_max_error_temp = nan

    def update(
        self, obs_dict: dict, next_obs_dict: dict, rewards_dict: dict, time_step: int
    ) -> None:
        for k in range(self.nb_agents):
            temp_error = (
                next_obs_dict[k]["indoor_temp"]
                - next_obs_dict[k]["target_temp"] / self.nb_agents
            )
            self.cumul_temp_offset += temp_error
            self.cumul_temp_error += np.abs(temp_error)
            self.max_temp_error = max(self.max_temp_error, temp_error)
            self.cumul_avg_reward += rewards_dict[k] / self.nb_agents

            if time_step >= self.start_stats_from:
                self.cumul_squared_error_temp += temp_error**2

            signal_error = (
                obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"]
            ) / (self.nb_agents**2)

            self.cumul_signal_offset += signal_error
            self.cumul_signal_error += np.abs(signal_error)

        self.cumul_OD_temp += obs_dict[0]["OD_temp"]
        self.cumul_signal += obs_dict[0]["reg_signal"]
        self.cumul_cons += obs_dict[0]["cluster_hvac_power"]

        if time_step >= self.start_stats_from:
            self.cumul_squared_error_sig += obs_dict[0]["reg_signal"] ** 2
            self.cumul_squared_max_error_temp = self.max_temp_error**2

    def log(self, time_step: int, time_steps_log: int, time: datetime):
        mean_avg_return = self.cumul_avg_reward / time_steps_log
        mean_temp_offset = self.cumul_temp_offset / time_steps_log
        mean_temp_error = self.cumul_temp_error / time_steps_log
        mean_signal_offset = self.cumul_signal_offset / time_steps_log
        mean_signal_error = self.cumul_signal_error / time_steps_log
        mean_OD_temp = self.cumul_OD_temp / time_steps_log
        mean_signal = self.cumul_signal / time_steps_log
        mean_consumption = self.cumul_cons / time_steps_log

        self.update_rms(time_step)

        if time_step >= self.start_stats_from:
            self.update_rms(time_step)
        else:
            self.rmse_sig_per_ag = nan
            self.rmse_temp = nan
            self.rms_max_error_temp = nan

        metrics = {
            "Mean train return": mean_avg_return,
            "Mean temperature offset": mean_temp_offset,
            "Mean temperature error": mean_temp_error,
            "Mean signal error": mean_signal_error,
            "Mean signal offset": mean_signal_offset,
            "Mean outside temperature": mean_OD_temp,
            "Mean signal": mean_signal,
            "Mean consumption": mean_consumption,
            "Time (hour)": time.hour,
            "Time step": time_step,
        }

        self.wandb_service.log(metrics)

        logger.info(f"Stats : {metrics}")

        return metrics

    def update_final(self) -> None:
        self.update_rms(self.nb_time_steps)
        rms_log = {
            "RMSE signal per agent": self.rmse_sig_per_ag,
            "RMSE temperature": self.rmse_temp,
            "RMS Max Error temperature": self.rms_max_error_temp,
        }

        self.wandb_service.log(rms_log)

        logger.info(f"RMSE Signal per agent: {self.rmse_sig_per_ag} W")
        logger.info(f"RMSE Temperature: {self.rmse_temp} C")
        logger.info(f"RMS Max Error Temperature: {self.rms_max_error_temp} C")

    def reset(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.max_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        self.cumul_OD_temp = 0
        self.cumul_signal = 0
        self.cumul_cons = 0

    def update_rms(self, time_step: int) -> None:
        self.rmse_sig_per_ag = (
            np.sqrt(self.cumul_squared_error_sig / (time_step - self.start_stats_from))
            / self.nb_agents
        )
        self.rmse_temp = np.sqrt(
            self.cumul_squared_error_temp
            / ((time_step - self.start_stats_from) * self.nb_agents)
        )
        self.rms_max_error_temp = np.sqrt(
            self.cumul_squared_max_error_temp / (time_step - self.start_stats_from)
        )

    def log_test_results(
        self,
        mean_avg_return: float,
        mean_temp_error: float,
        mean_signal_error: float,
        tr_time_steps: int,
    ) -> None:
        log = {
            "Mean test return": mean_avg_return,
            "Test mean temperature error": mean_temp_error,
            "Test mean signal error": mean_signal_error,
            "Training steps": tr_time_steps,
        }
        self.wandb_service.log(log)

    def save_actor(self, path: str) -> None:
        self.wandb_service.save(path)

from cmath import nan
from datetime import datetime
from typing import Dict

import numpy as np

from app.core.environment.environment import EnvironmentObsDict
from app.services.wandb_service import WandbManager
from app.utils.logger import logger


class Metrics:
    """
    A class that keeps track of various metrics values during training and logs them using a WandbManager instance.

    Attributes:
        wandb_service : WandbManager
            An instance of WandbManager used for logging metrics.
        nb_time_steps : int
            Total number of time steps.
        start_stats_from : int
            Time step from which to start computing statistics.
        nb_agents : int
            Number of agents in the environment.
        cumul_avg_reward : float
            Cumulative average reward over all time steps.
        cumul_temp_offset : float
            Cumulative temperature offset over all time steps.
        cumul_temp_error : float
            Cumulative temperature error over all time steps.
        cumul_signal_offset : float
            Cumulative signal offset over all time steps.
        cumul_signal_error : float
            Cumulative signal error over all time steps.
        cumul_squared_error_temp : float
            Cumulative squared error of temperature over all time steps.
        max_temp_error : float
            Maximum temperature error over all time steps.
        cumul_OD_temp : float
            Cumulative outside temperature over all time steps.
        cumul_signal : float
            Cumulative signal over all time steps.
        cumul_cons : float
            Cumulative consumption over all time steps.
        cumul_squared_error_sig : float
            Cumulative squared error of signal over all time steps.
        rmse_sig_per_ag : float
            Root-mean-squared error of signal per agent.
        rmse_temp : float
            Root-mean-squared error of temperature.
        rms_max_error_temp : float
            Root-mean-squared maximum error of temperature.
    """

    def __init__(
        self,
        wandb_service: WandbManager,
    ) -> None:
        """
        Initializes the Metrics class with a WandbManager instance.

        Parameters:
        wandb_service : WandbManager
            Instance of WandbManager to use for logging metrics.

        Returns:
            None
        """
        self.wandb_service = wandb_service

    def initialize(
        self, nb_agents: int, start_stats_from: int, nb_time_steps: int
    ) -> None:
        """
        Initializes various metrics values to zero.

        Parameters:
            nb_agents : int
                Number of agents in the environment.
            start_stats_from : int
                Time step from which to start computing statistics.
            nb_time_steps : int
                Total number of time steps.

        Returns:
            None
        """
        self.nb_time_steps = nb_time_steps
        self.start_stats_from = start_stats_from
        self.nb_agents = nb_agents
        self.cumul_avg_reward = 0.0
        self.cumul_temp_offset = 0.0
        self.cumul_temp_error = 0.0
        self.cumul_signal_offset = 0.0
        self.cumul_signal_error = 0.0
        self.cumul_squared_error_temp = 0.0
        self.max_temp_error = 0.0
        self.cumul_OD_temp = 0.0
        self.cumul_signal = 0.0
        self.cumul_cons = 0.0
        self.cumul_squared_error_sig = 0.0
        self.rmse_sig_per_ag = nan
        self.rmse_temp = nan
        self.rms_max_error_temp = nan
        self.wandb_service.initialize()

    def update(
        self,
        obs_dict: Dict[int, EnvironmentObsDict],
        next_obs_dict: Dict[int, EnvironmentObsDict],
        rewards_dict: dict,
        time_step: int,
    ) -> None:
        """
        Updates the various metrics values for the given time step.

        Parameters:
            obs_dict : Dict[int, EnvironmentObsDict]
                Dictionary of observation values for the current time step.
            next_obs_dict : Dict[int, EnvironmentObsDict]
                Dictionary of observation values for the next time step.
            rewards_dict : dict
                Dictionary of rewards for the current time step.
            time_step : int
                The current time step.

        Returns:
            None
        """
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
        """
        Logs the various metrics values for the given time step.

        Parameters:

            time_step: The current time step.
            time_steps_log: The number of time steps to log.
            time: A datetime object indicating the current time.
        Returns:
            A dictionary of the logged metrics.
        """

        if time_step >= self.start_stats_from:
            self.update_rms(time_step)
        else:
            self.rmse_sig_per_ag = nan
            self.rmse_temp = nan
            self.rms_max_error_temp = nan

        metrics = {
            "Mean train return": self.cumul_avg_reward / time_steps_log,
            "Mean temperature offset": self.cumul_temp_offset / time_steps_log,
            "Mean temperature error": self.cumul_temp_error / time_steps_log,
            "Mean signal error": self.cumul_signal_offset / time_steps_log,
            "Mean signal offset": self.cumul_signal_offset / time_steps_log,
            "Mean outside temperature": self.cumul_OD_temp / time_steps_log,
            "Mean signal": self.cumul_signal / time_steps_log,
            "Mean consumption": self.cumul_cons / time_steps_log,
            "Time (hour)": time.hour,
            "Time step": time_step,
        }
        breakpoint()
        self.wandb_service.log(metrics)

        logger.info(f"Stats : {metrics}")

        return metrics

    def update_final(self) -> None:
        """
        Updates the various metrics values for the final time step.

        Parameters:
            None

        Returns:
            None
        """
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
        """
        Resets the cumulative average reward, temperature offset, temperature error,
        maximum temperature error, signal offset, signal error, OD temperature, signal,
        and consumption for all agents to zero.
        """
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
        """
        Updates the root-mean-squared errors for the signal and temperature for all agents
        based on the current cumulative squared errors and number of time steps since starting
        to collect statistics. The root-mean-squared error values are normalized by the number
        of agents.

        Parameters:
            time_step (int): The current time step of the simulation.
        """
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
        """
        Logs the mean test return, mean temperature error, mean signal error, and number of
        training steps to the WandB service.

        Parameters:
            mean_avg_return (float): The mean test return.
            mean_temp_error (float): The mean temperature error.
            mean_signal_error (float): The mean signal error.
            tr_time_steps (int): The number of training steps.
        """
        log = {
            "Mean test return": mean_avg_return,
            "Test mean temperature error": mean_temp_error,
            "Test mean signal error": mean_signal_error,
            "Training steps": tr_time_steps,
        }
        self.wandb_service.log(log)

    def save_actor(self, path: str) -> None:
        """
        Saves the actor network to the specified path using the WandB service.

        Parameters:
            path (str): The path to save the actor network.
        """
        self.wandb_service.save(path)

from copy import deepcopy
import pandas as pd

from app.core.agents.controllers.controller import Controller

global_myopic_memory = [None, None]


class GreedyMyopic(Controller):
    """
    A controller that tries to distribute the energy budget among all the agents prioritizing the agents with a high
    temperature compared to the target temperature in a greedy way.

    Attributes:
        agent_properties (dict): A dictionary containing information about the agent.
        id (int): The ID of the agent.
        last_obs (pandas.DataFrame): The last observation received by the agent.
        time_step (int): The current time step.

    """

    actions_df = []

    def __init__(self, agent_properties, config_dict, num_state=None) -> None:
        """
        Initialize the GreedyMyopic object.

        Parameters:
            - agent_properties (dict): A dictionary containing the properties of the agent.
            - config_dict (dict): A dictionary containing the configuration parameters.
            - num_state (int): The number of states. Default is None.

        """
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]
        self.last_obs = pd.DataFrame(
            columns=(
                "temperature_difference",
                "power_consumption",
                "lockout",
                "reg_signal",
            )
        )
        self.time_step = 0

    def act(self, obs):
        """
        Returns the action to be taken by the agent.

        Parameter:
            obs (list): A list containing the observations of the agent.

        Returns:
            int: The action to be taken by the agent.
        """
        self.time_step += 1
        obs_dict = deepcopy(obs)
        if global_myopic_memory[0] != self.time_step:
            self.last_obs = obs_dict
            self.get_action(obs_dict)
            global_myopic_memory[0] = self.time_step
            global_myopic_memory[1] = GreedyMyopic.actions_df
        action = global_myopic_memory[1].loc[self.id]["HVAC_status"]

        return action

    def get_action(self, obs) -> None:
        """
        Compute the action to be taken by the agent based on the observations.

        Parameter:
            obs (list): A list containing the observations of the agent.
        Returns:
            None

        """
        obs = pd.DataFrame(obs).transpose()
        obs["temperature_difference"] = -(obs["indoor_temp"] - obs["target_temp"])

        obs["power_consumption"] = obs["cooling_capacity"] / obs["cop"]
        obs = obs[
            [
                "temperature_difference",
                "power_consumption",
                "lockout",
                "reg_signal",
            ]
        ]
        obs.sort_values("temperature_difference", inplace=True)
        obs["HVAC_status"] = 0
        target = obs["reg_signal"][0]
        total_consumption = 0

        for index, row in obs.iterrows():
            if (
                row["power_consumption"] + total_consumption < target
                or abs(row["power_consumption"] + total_consumption - target)
                < abs(total_consumption - target)
                and not row["lockout"]
            ):
                total_consumption += row["power_consumption"]
                obs.at[index, "HVAC_status"] = 1

        GreedyMyopic.actions_df = obs

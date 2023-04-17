import pandas as pd

global_myopic_memory = [None, None]


class GreedyMyopic:
    """Try to distribute the energy budget among all the agents prioritizing the agents
    with a high temperature compared to the target temperature in a greedy way"""

    actions_df = []

    def __init__(self, agent_properties, config_dict, num_state=None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]
        self.last_obs = pd.DataFrame(
            columns=(
                "temperature_difference",
                "power_consumption",
                "hvac_lockout",
                "reg_signal",
            )
        )
        self.time_step = 0

    def act(self, obs):
        self.time_step += 1
        if global_myopic_memory[0] != self.time_step:
            self.last_obs = obs
            self.get_action(obs)
            global_myopic_memory[0] = self.time_step
            global_myopic_memory[1] = GreedyMyopic.actions_df
        action = global_myopic_memory[1].loc[self.id]["HVAC_status"]

        return action

    def get_action(self, obs):
        obs = pd.DataFrame(obs).transpose()
        obs["temperature_difference"] = -(obs["indoor_temp"] - obs["target_temp"])

        obs["power_consumption"] = obs["cooling_capacity"] / obs["COP"]
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

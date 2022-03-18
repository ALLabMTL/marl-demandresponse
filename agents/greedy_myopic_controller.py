import pandas as pd

from torch import Tensor


class GreedyMyopic(object):
    """ Not really a bang bang controller but: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    actions_df = []

    def __init__(self, agent_properties, config_dict):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]
        self.last_obs = pd.DataFrame(columns=(
            "temperature_difference", "power_consumption", "hvac_lockout",  "reg_signal"))

    def act(self, obs):
        if (self.id == "0_1"):
            self.last_obs = obs
            self.get_action(obs)

        action = GreedyMyopic.actions_df.loc[self.id]["HVAC_status"]

        return action

    def get_action(self, obs):
        obs = pd.DataFrame(obs).transpose()
        obs["temperature_difference"] = -(obs["house_temp"] -
                                          obs["house_target_temp"])

        obs["power_consumption"] = obs["hvac_cooling_capacity"] / \
            obs["hvac_COP"]
        obs = obs[[
            "temperature_difference", "power_consumption", "hvac_lockout",  "reg_signal"]]
        obs.sort_values("temperature_difference", inplace=True)
        obs["HVAC_status"] = 0
        target = obs["reg_signal"][0]
        total_consumption = 0

        for index, row in obs.iterrows():
            if (row["power_consumption"] + total_consumption < target
                    or abs(row["power_consumption"] + total_consumption - target) < abs(total_consumption-target)
                    and not row["hvac_lockout"]):
                total_consumption += row["power_consumption"]
                obs.at[index, "HVAC_status"] = 1

        GreedyMyopic.actions_df = obs

from typing import Literal

from app.core.agents.controllers.controller import Controller


class AlwaysOnController(Controller):
    """Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict, num_state=None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs_dict) -> Literal[True]:
        return True


class DeadbandBangBangController(Controller):
    """Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict, num_state=None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        house_temp = obs_dict["indoor_temp"]
        house_target_temp = obs_dict["target_temp"]
        house_deadband = obs_dict["deadband"]
        hvac_turned_on = obs_dict["turned_on"]

        if house_temp < house_target_temp - house_deadband / 2:
            action = False
            # print("Too cold!")

        elif house_temp > house_target_temp + house_deadband / 2:
            action = True
            # print("Too hot!")
        else:
            action = hvac_turned_on

        return action


class BangBangController(Controller):
    """
    Cools when temperature is hotter than target (no interest for deadband). Limited on the hardware side by lockout (but does not know about it)
    """

    def __init__(self, agent_properties, config_dict, num_state=None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs_dict) -> bool:
        obs_dict = obs_dict[self.id]
        house_temp = obs_dict["indoor_temp"]
        house_target_temp = obs_dict["target_temp"]

        if house_temp <= house_target_temp:
            action = False

        elif house_temp > house_target_temp:
            action = True

        return action


class BasicController(Controller):
    """Not really a bang bang controller but: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict, num_state=None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs_dict) -> bool:
        obs_dict = obs_dict[self.id]
        house_temp = obs_dict["indoor_temp"]
        house_target_temp = obs_dict["target_temp"]
        house_deadband = obs_dict["deadband"]
        hvac_turned_on = obs_dict["turned_on"]

        if house_temp < house_target_temp - house_deadband / 2:
            action = False
        elif house_temp > house_target_temp + house_deadband / 2:
            action = True
        else:
            action = hvac_turned_on

        return action

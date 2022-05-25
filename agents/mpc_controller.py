from distutils.log import error


class MPCController(object):
    """Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

        self.mode = config_dict["default_env_prop"]["power_grid_prop"]["signal_mode"]

        if self.mode == "sinusoidals":
            self.signal_parameter = config_dict["default_env_prop"]["power_grid_prop"][
                "signal_parameters"
            ]["sinusoidals"]

    def agent_transition():
        pass

    def environnement_transition():
        pass

    def act(self, obs):
        pass

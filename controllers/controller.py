"""Base class for all controllers."""

import abc


class Controller(abc.ABC):
    """Base class for all controllers."""

    def __init__(self, agent_properties, config_dict, num_state=None) -> None:
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    @abc.abstractmethod
    def run(self) -> None:
        """Run the controller."""

    @abc.abstractmethod
    def act(self, obs) -> bool:
        """Return the action to take given the observation."""

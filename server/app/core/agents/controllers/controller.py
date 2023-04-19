import abc
from typing import Dict, Union


class Controller(abc.ABC):
    """
    Abstract base class for implementing controllers in the environment.

    Attributes:
        num_state (int): The number of states in the environment. Default is 22.
        num_action (int): The number of actions available to the controller. Default is 2.

    """

    @abc.abstractmethod
    def __init__(
        self, agent_properties, config_dict, num_state=22, num_action=2
    ) -> None:
        """
        Initialize the controller object.

        Parameters:
            - agent_properties: dict, a dictionary of agent properties.
            - config_dict: dict, a dictionary of configuration parameters.
            - num_state: int, the number of states in the agent's observation space. Defaults to 22.
            - num_action: int, the number of actions in the agent's action space. Defaults to 2.

        Returns:
            None
        """

    @abc.abstractmethod
    def act(self, obs_dict: dict) -> Union[bool, Dict[str, bool]]:
        """
        This method takes in the current observation of the environment and returns the action to take.

        Parameters:
            obs_dict: dict, a dictionary containing the current observation of the environment.

        Returns:
            action: Union[bool, Dict[str, bool]], the action to take. If the action is binary, then it will be a single boolean
            value. If the action is a dictionary, then it will contain boolean values for each key-value pair representing the
            actions for each subcomponent of the agent.
        """

import abc

from typing import Union, Dict


class Controller(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, agent_properties, config_dict, num_state=22, num_action=2
    ) -> None:
        pass

    @abc.abstractmethod
    def act(self, obs_dict: dict) -> Union[bool, Dict[str, bool]]:
        pass

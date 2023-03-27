from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def act(self, obs_dict: dict) -> dict:
        pass

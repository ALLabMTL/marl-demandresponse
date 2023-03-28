from abc import ABC, abstractmethod


class Trainable(ABC):
    @abstractmethod
    def select_action(self, obs_dict: dict) -> dict:
        pass

    @abstractmethod
    def store_transition(
        self,
        obs_dict: dict,
        next_obs_dict: dict,
        action: dict,
        action_prob: dict,
        rewards_dict: dict,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def update(self, time_step: int) -> None:
        pass

    def save(self, path, time_step=None) -> None:
        pass

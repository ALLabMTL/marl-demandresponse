from abc import ABC, abstractmethod


class Agent(ABC):
    buffer: dict
    batch_size: int

    @abstractmethod
    def select_action(self, obs_dict: dict) -> dict:
        pass

    @abstractmethod
    def store_transition(self, transition, agent) -> dict:
        pass

    @abstractmethod
    def update(self, time_step: int) -> None:
        pass


class Trainable(ABC):
    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    def test(self, env, config_dict, opt, tr_time_step, num_episodes=10):
        pass

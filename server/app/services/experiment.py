from abc import ABC, abstractmethod

from app.services.parser_service import MarlConfig


class Experiment(ABC):
    speed: float
    agent_name: str
    stop: bool
    pause: bool

    @abstractmethod
    def initialize(self, config: MarlConfig) -> None:
        pass

    @abstractmethod
    async def start(self, config: MarlConfig) -> None:
        pass

    @abstractmethod
    async def stop_sim(self, stop_state: bool)-> None:
        pass

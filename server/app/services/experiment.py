from abc import ABC, abstractmethod


class Experiment(ABC):
    speed: float
    agent_name: str
    stop: bool
    pause: bool

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

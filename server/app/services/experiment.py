from abc import ABC, abstractmethod


class Experiment(ABC):
    speed: float

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

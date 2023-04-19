from abc import ABC, abstractmethod

from app.services.parser_service import MarlConfig


class Experiment(ABC):
    """
    The Experiment class is an abstract base class that defines the structure of an experiment for a multi-agent reinforcement learning problem. It has four attributes:

    Attributes:
        speed: a float value representing the speed at which the simulation runs.
        agent_name: a string representing the name of the agent used in the experiment.
        stop: a boolean indicating whether or not the experiment should be stopped.
        pause: a boolean indicating whether or not the experiment should be paused.
    """

    speed: float
    agent_name: str
    stop: bool
    pause: bool

    @abstractmethod
    async def initialize(self, config: MarlConfig) -> None:
        """It's an abstract method that initializes the experiment with the given MarlConfig object."""
        pass

    @abstractmethod
    async def start(self, config: MarlConfig) -> None:
        """Tt's an abstract method that starts the experiment with the given MarlConfig object."""
        pass

    @abstractmethod
    async def stop_sim(self, stop_state: bool) -> None:
        """It's an abstract method that stops the simulation and sets the stop attribute to stop_state."""
        pass

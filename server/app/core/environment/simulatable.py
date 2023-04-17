from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")
U = TypeVar("U")


class Simulatable(ABC):
    """
    Define a set of methods that should be implemented by any class that models a simulation that can be reset, stepped through, and observed.

    Attributes:
        init_props: A Pydantic BaseModel representing the initial properties of the simulation.
    """

    init_props: BaseModel

    @abstractmethod
    def reset(self):
        """Reset the simulation to its initial state."""
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        """Take a step in the simulation using the provided arguments."""
        pass

    @abstractmethod
    def get_obs(self):
        """Get the current observation of the simulation."""
        pass

    @abstractmethod
    def apply_noise(self) -> None:
        """Apply noise to the simulation to simulate real-world variability. This method does not return anything."""
        pass

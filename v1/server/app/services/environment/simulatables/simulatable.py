from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Simulatable(ABC):
    initial_properties: BaseModel

    @abstractmethod
    def step() -> None:
        pass

    @abstractmethod
    def apply_noise(noise_properties: T) -> None:
        pass

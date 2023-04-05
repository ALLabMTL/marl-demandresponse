from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")
U = TypeVar("U")


class Simulatable(ABC):
    initial_properties: BaseModel
    noise_properties: BaseModel

    @abstractmethod
    def reset(self) -> dict:
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def get_obs(self) -> dict:
        pass

    @abstractmethod
    def apply_noise(self) -> None:
        pass

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")
U = TypeVar("U")


class Simulatable(ABC):
    initial_properties: BaseModel
    noise_properties: BaseModel

    @abstractmethod
    def _reset(self) -> dict:
        pass

    @abstractmethod
    def _step(self) -> dict:
        pass

    @abstractmethod
    def _get_obs(self) -> dict:
        pass

    @abstractmethod
    def apply_noise(self) -> None:
        pass

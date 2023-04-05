from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")
U = TypeVar("U")


class Simulatable(ABC):
    init_props: BaseModel

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_obs(self):
        pass

    @abstractmethod
    def apply_noise(self) -> None:
        pass

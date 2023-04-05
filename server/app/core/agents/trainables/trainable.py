from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from pydantic import BaseModel

from app.core.environment.environment_properties import EnvironmentObsDict


class Trainable(ABC):
    @abstractmethod
    def __init__(self, config: BaseModel, num_state=22, num_action=2, seed=1) -> None:
        pass

    @abstractmethod
    def select_actions(self, observations: List[np.ndarray]) -> Dict[int, bool]:
        pass

    @abstractmethod
    def store_transition(
        self,
        observations: List[np.ndarray],
        next_observations: List[np.ndarray],
        rewards: Dict[int, float],
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def update(self, time_step: int) -> None:
        pass

    def save(self, path, time_step=None) -> None:
        pass

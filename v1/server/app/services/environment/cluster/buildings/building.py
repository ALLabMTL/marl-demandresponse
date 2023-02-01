from abc import ABC
from typing import List, TypeVar

from v1.server.app.services.environment.simulatables.simulatable import Simulatable

T = TypeVar("T")
U = TypeVar("U")


class Building(ABC):
    id: int = 1
    building_properties: T
    noise_properties: U
    current_temperature: float
    current_mass_temperature: float
    simulated_objects: List[Simulatable]  # Pour l'instant on peut juste mettre hvac

    def __init__(self) -> None:
        self.set_building_properties()

    def step() -> None:
        pass

    def update_temperature() -> None:
        pass

    def apply_noise(self) -> None:
        pass

    def set_building_properties() -> None:
        # try:
        # parser_service.parse_building_properties()
        # except:
        # logger.error
        pass

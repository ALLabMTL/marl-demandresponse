from app.services.controller_manager import ControllerManager
from app.services.training_manager import TrainingManager
from app.services.experiment import Experiment


class ExperimentManager:
    experiment: Experiment

    def __init__(
        self, controller_manager: ControllerManager, training_manager: TrainingManager
    ) -> None:
        self.controller_manager = controller_manager
        self.training_manager = training_manager
        self.experiment = controller_manager

    def initialize(self, mode="simulation") -> None:
        if mode == "train":
            self.experiment = self.controller_manager
        elif mode == "simulation":
            self.experiment = self.training_manager

        self.experiment.initialize()

    async def start(self) -> None:
        await self.experiment.start()

    def change_speed(self, speed: float) -> None:
        self.experiment.speed = speed

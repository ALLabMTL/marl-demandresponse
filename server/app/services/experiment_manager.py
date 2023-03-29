from app.services.controller_manager import ControllerManager
from app.services.experiment import Experiment
from app.services.parser_service import ParserService
from app.services.training_manager import TrainingManager


class ExperimentManager:
    experiment: Experiment

    def __init__(
        self,
        controller_manager: ControllerManager,
        training_manager: TrainingManager,
        parser_service: ParserService,
    ) -> None:
        self.controller_manager = controller_manager
        self.training_manager = training_manager
        self.experiment = controller_manager
        self.parser_service = parser_service

    def initialize(self, mode="simulation") -> None:
        self.parser_service.config.CLI_config.mode = mode
        if mode == "train":
            self.experiment = self.controller_manager
        elif mode == "simulation":
            self.experiment = self.training_manager

        self.experiment.initialize()

    async def start(self) -> None:
        await self.experiment.start()

    def change_speed(self, speed: float) -> None:
        self.experiment.speed = speed

    def update_experiment_state(self, stop: bool) -> None:
        self.experiment.stop = stop

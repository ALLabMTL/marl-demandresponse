from app.services.controller_manager import ControllerManager
from app.services.experiment import Experiment
from app.services.parser_service import MarlConfig, ParserService
from app.services.training_manager import TrainingManager


class ExperimentManager:
    experiment: Experiment

    def __init__(
        self,
        controller_manager: ControllerManager,
        training_manager: TrainingManager,
    ) -> None:
        self.controller_manager = controller_manager
        self.training_manager = training_manager
        self.experiment = training_manager
        self.parser = ParserService()
        self.config = self.parser.config
        if self.config.simulation_props.mode == "train":
            self.experiment = self.training_manager
        elif self.config.simulation_props.mode == "simulation":
            self.experiment = self.controller_manager

    def initialize(self) -> None:
        self.experiment.initialize(self.config)

    async def start(self) -> None:
        await self.experiment.start(self.config)

    def change_speed(self, speed: float) -> None:
        self.experiment.speed = speed

    def update_experiment_state(self, stop: bool) -> None:
        self.experiment.stop = stop

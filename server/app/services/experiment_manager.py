from app.services.controller_manager import ControllerManager
from app.services.experiment import Experiment
from app.services.parser_service import MarlConfig, ParserService
from app.services.training_manager import TrainingManager
from app.utils.logger import logger


class ExperimentManager:
    """
    Class that manages the experiment, including initializing it, starting it, and updating its state.

    Attributes:
        experiment (Experiment): The current Experiment instance.
        config (MarlConfig): Config of experiment.
    """

    experiment: Experiment
    config: MarlConfig

    def __init__(
        self,
        controller_manager: ControllerManager,
        training_manager: TrainingManager,
    ) -> None:
        """
        Initialize a new instance of the ExperimentManager class.

        Parameters:
            controller_manager: A ControllerManager object used to run the simulation in "simulation" mode.
            training_manager: A TrainingManager object used to run the simulation in "train" mode.
        """
        self.controller_manager = controller_manager
        self.training_manager = training_manager
        self.experiment = training_manager

    async def initialize(self) -> None:
        """Initialize the experiment."""
        parser = ParserService()
        self.config = parser.config
        logger.debug("Config initialized")
        if self.config.simulation_props.mode == "train":
            self.experiment = self.training_manager
        elif self.config.simulation_props.mode == "simulation":
            self.experiment = self.controller_manager
        self.experiment.stop = False

    async def start(self) -> None:
        """Start the experiment."""
        await self.experiment.start(self.config)

    def change_speed(self, speed: float) -> None:
        """
        Change the speed of the simulation.

        Parameters:
            speed: A float value representing the new speed of the simulation.
        """
        self.experiment.speed = speed

    async def update_experiment_state(self, stop: bool) -> None:
        """
        Update the state of the experiment (i.e. whether it is stopped or running).

        Parameters:
            stop: A boolean value representing whether the experiment should be stopped or not.
        """
        self.experiment.stop = stop
        await self.experiment.stop_sim(stop)

    def pause_simulation(self) -> None:
        """Pause the simulation."""
        self.experiment.pause = True

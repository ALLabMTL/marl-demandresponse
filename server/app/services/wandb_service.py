import wandb
from app.services.parser_service import ParserService

metrics = [
    "Mean train return",
    "Mean temperature offset",
    "Mean temperature error",
    "Mean signal offset",
    "Mean signal error",
    "Mean next signal error",
    "Mean next signal offset",
    "Mean test return",
    "Test mean temperature error",
    "Test mean signal error",
]


class WandbManager:
    """
    Class that manages logging to Weights and Biases (wandb) platform.

    Attributes:
        should_log (bool): Determines if logging is enabled.
    """

    should_log: bool = False

    def initialize(self) -> None:
        """
        Initialize the logging manager with the provided configuration.

        Parameters:
            log (bool): Determines if logging is enabled. Default is False.
        """
        global_conf = ParserService().config
        self.should_log = global_conf.CLI_config.wandb
        if self.should_log:
            log_config = {"config_file": global_conf}
            self.wandb_run = wandb.init(
                settings=wandb.Settings(start_method="spawn"),
                project=global_conf.CLI_config.wandb_project,
                config=log_config,
                name=f"{global_conf.CLI_config.experiment_name}_TCLs-{global_conf.env_prop.cluster_prop.nb_agents}_netseed-{global_conf.simulation_props.net_seed}",
            )
            for metric in metrics:
                self.wandb_run.define_metric(name=metric, step_metric="Training steps")

    def log(self, data: dict) -> None:
        """
        Log the provided data to wandb if logging is enabled.

        Parameters:
            data (dict): Dictionary of data to log.
        """
        if self.should_log:
            self.wandb_run.log(data)

    def save(self, path) -> None:
        """
        Save the provided path to wandb if logging is enabled.

        Parameters:
            path (str): Path to save.
        """
        if self.should_log:
            wandb.save(path)

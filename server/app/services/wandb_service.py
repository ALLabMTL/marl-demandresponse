import wandb

# from wandb import Run, RunDisabled
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

    def initialize(
        self, config_dict, env_seed, net_seed, exp_name, nb_agents, log=False
    ) -> None:
        """
        Initialize the logging manager with the provided configuration.

        Parameters:
            config_dict (dict): Configuration dictionary.
            env_seed (int): Environment seed value.
            net_seed (int): Network seed value.
            exp_name (str): Experiment name.
            nb_agents (int): Number of agents.
            log (bool): Determines if logging is enabled. Default is False.
        """
        self.should_log = log
        if log:
            log_config = {"config_file": config_dict}
            self.wandb_run = wandb.init(
                settings=wandb.Settings(start_method="fork"),
                project="ProofConcept",
                entity="marl-dr",
                config=log_config,
                name=f"{exp_name}_TCLs-{nb_agents}_envseed-{env_seed}_netseed-{net_seed}",
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

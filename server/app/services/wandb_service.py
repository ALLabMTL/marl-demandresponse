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
    should_log: bool = False

    def initialize(
        self, config_dict, env_seed, net_seed, exp_name, nb_agents, log=False
    ) -> None:
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
        if self.should_log:
            self.wandb_run.log(data)

    def save(self, path) -> None:
        if self.should_log:
            wandb.save(path)

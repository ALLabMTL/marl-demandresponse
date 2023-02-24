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
    def init(self, opt, config_dict):
        log_config = {"opt": vars(opt), "config_file": config_dict}
        self.wandb_run = wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project="ProofConcept",
            entity="marl-dr",
            config=log_config,
            name=f"{opt.exp}_TCLs-{opt.nb_agents}_envseed-{opt.env_seed}_netseed-{opt.net_seed}",
        )
        for metric in metrics:
            self.wandb_run.define_metric(name=metric, step_metric="Training steps")

    def log(self, data: dict) -> None:
        self.wandb_run.log(data)

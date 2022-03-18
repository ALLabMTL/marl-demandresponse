import wandb

wandb.init(project="tutorial", entity="philippemaisonneuve")

for i in range(100):

    wandb.log({"loss": 1})
    wandb.log({"x": i})

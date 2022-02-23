import wandb



def wandb_setup(opt):

    wandb_run = wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="ProofConcept",
        entity="marl-dr",
        config=opt,
        name="%s_TCLs-%d_envseed-%d_netseed-%d"%(opt.exp, opt.nb_agents, opt.env_seed, opt.net_seed),
    )

    wandb_run.define_metric(name='Mean train return', step_metric='Training steps')
    wandb_run.define_metric(name='Mean temperature offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean test return', step_metric='Training steps')

    return wandb_run
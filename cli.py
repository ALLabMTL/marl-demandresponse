
import argparse

def cli_train():
    parser = argparse.ArgumentParser(description="Training options")

    parser.add_argument(
        "--nb_agents",
        type=int,
        default=-1,
        help="Number of agents (TCLs)",
    )

    parser.add_argument(
        "--nb_tr_episodes",
        type=int,
        default=1000,
        help="Number of episodes (environment resets) for training",
    )

    parser.add_argument(
        "--nb_tr_epochs",
        type=int,
        default=200,
        help="Number of epochs (policy updates) for training",
    )

    parser.add_argument(
        "--nb_tr_logs",
        type=int,
        default=100,
        help="Number of logging points for training stats",
    )

    parser.add_argument(
        "--nb_test_logs",
        type=int,
        default=10,
        help="Number of logging points for testing stats (and thus, testing sessions)",
    )

    parser.add_argument(
        "--nb_time_steps",
        type=int,
        default=1000000,
        help="Total number of time steps",
    )

    parser.add_argument(
        "--nb_time_steps_test",
        type=int,
        default=1000,
        help="Total number of time steps in an episode at test time",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size",
    )

    parser.add_argument(
        "--net_seed",
        type=int,
        default=1,
        help="Neural network seed",
    )

    parser.add_argument(
        "--env_seed",
        type=int,
        default=1,
        help="Environment seed",
    )

    parser.add_argument(
        "--exp",
        type=str,
        #required=True,
        help="Experiment name",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Add to prevent logging to wandb",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Add to generate a visual render of the simulation",
    )

    parser.add_argument(
        "--render_after",
        type=int,
        default=-1,
        help="Delay in time steps before rendering")

    parser.add_argument(
        "--cooling_capacity",
        type=int,
        default=-1,
        help="Default cooling capacity of the HVACs",
    )

    parser.add_argument(
        "--time_step",
        type=int,
        default=-1,
        help="Time step in seconds",
    )

    parser.add_argument(
        "--lockout_duration",
        type=int,
        default=-1,
        help="Default AC lockout duration, in seconds",
    )

    parser.add_argument(
        "--save_actor_name",
        type=str,
        default="dqn", # "none"
        help="Name to store the actor agent after training",
    )

    parser.add_argument(
        "--exploration_temp",
        type=float,
        default=1.0,
        help="Temperature of the policy softmax. Higher temp -> more exploration."
    )

    parser.add_argument(
        "--signal_mode",
        type=str,
        default="config",
        help="Mode of power grid regulation signal simulation."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=-1,
        help="Tradeoff parameter for loss function: temperature penalty + alpha * regulation signal penalty."
    )

    opt, _ = parser.parse_known_args()
    
    return opt

import argparse

def cli_train():
    parser = argparse.ArgumentParser(description="Training options")

    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        help="Type of agent (dqn, ppo)",
    )

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
        default=20,
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
        default=100,
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
        default=50000,
        help="Total number of time steps in an episode at test time",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        required=True,
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
        default=None,
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
        help="Mode of the noise on the power grid regulation signal simulation. Choices: [none, regular_steps, sinusoidals, config]"
    )

    parser.add_argument(
        "--base_power_mode",
        type=str,
        default="config",
        help="Mode for the base (low frequency) regulation signal simulation. Choices: [constant, interpolation, config]"
    )

    parser.add_argument(
        "--alpha_temp",
        type=float,
        default=-1,
         help="Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty."
    )

    parser.add_argument(
        "--alpha_sig",
        type=float,
        default=-1,
        help="Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty."
    )

    parser.add_argument(
        "--house_noise_mode",
        type=str,
        default="config",
        help="Mode of noise over house parameters.")

    parser.add_argument(
        "--house_noise_mode_test",
        type=str,
        default="train",
        help="Mode of noise over house parameters for test environment.")

    parser.add_argument(
        "--hvac_noise_mode",
        type=str,
        default="config",
        help="Mode of noise over HVAC parameters.")

    parser.add_argument(
        "--hvac_noise_mode_test",
        type=str,
        default="train",
        help="Mode of noise over HVAC parameters for test environment.")

    parser.add_argument(
        "--OD_temp_mode",
        type=str,
        default="config",
        help="Mode of outdoors temperature.")

    parser.add_argument(
        "--no_solar_gain",
        action="store_true",
        help="Removes the solar gain from the simulation.")

    opt = parser.parse_args()
    
    return opt


def cli_deploy(agents_dict):
    parser = argparse.ArgumentParser(description="Deployment options")
    
    parser.add_argument(
        "--base_power_mode",
        type=str,
        default="config",
        help="Mode for the base (low frequency) regulation signal simulation. Choices: [constant, interpolation, config]"
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=agents_dict.keys(),
        required=True,
        help="Agent for control",
    )

    parser.add_argument(
        "--nb_agents",
        type=int,
        default=1,
        help="Number of agents (TCLs)",
    )

    parser.add_argument(
        "--nb_time_steps",
        type=int,
        default=1000,
        help="Number of time steps in an episode",
    )

    parser.add_argument(
        "--nb_logs",
        type=int,
        default=100,
        help="Number of logging points for training stats",
    )

    parser.add_argument(
        "--env_seed",
        type=int,
        default=1,
        help="Environment seed",
    )

    parser.add_argument(
        "--net_seed",
        type=int,
        default=1,
        help="Network and torch seed",
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="Deploy",
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
        "--actor_name", type=str, default=None, help="Name of the trained agent to load"
    )

    parser.add_argument(
        "--exploration_temp",
        type=float,
        default=1.0,
        help="Temperature of the policy softmax. Higher temp -> more exploration.",
    )

    parser.add_argument(
        "--signal_mode",
        type=str,
        default="config",
        help="Mode of power grid regulation signal simulation.",
    )

    parser.add_argument(
        "--house_noise_mode",
        type=str,
        default="config",
        help="Mode of noise over house parameters.",
    )

    parser.add_argument(
        "--hvac_noise_mode",
        type=str,
        default="config",
        help="Mode of noise over hvac parameters.",
    )

    parser.add_argument(
        "--OD_temp_mode", type=str, default="config", help="Mode of outdoors temperature."
    )

    parser.add_argument(
        "--no_solar_gain",
        action="store_true",
        help="Removes the solar gain from the simulation.",
    )

    parser.add_argument(
        "--MPC_rolling_horizon",
        type=int,
        default=-1,
        help="Duration of the MPC rooling horizon in time step",
    )

    opt = parser.parse_args()
    
    return opt
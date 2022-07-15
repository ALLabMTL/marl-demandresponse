import argparse


def cli_train():
    parser = argparse.ArgumentParser(description="Training options")


### Context

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
        help="Delay in time steps before rendering",
    )

    parser.add_argument(
        "--save_actor_name",
        type=str,
        default=None,
        help="Name to store the actor agent after training",
    )

    parser.add_argument(
        "--nb_inter_saving_actor",
        type = int,
        default = 0,
        help="Number of intermediate times the actor is saved during training."
        )

### Environment

    parser.add_argument(
        "--nb_agents",
        type=int,
        default=-1,
        help="Number of agents (TCLs)",
    )

    parser.add_argument(
        "--env_seed",
        type=int,
        default=1,
        help="Environment seed",
    )

    parser.add_argument(
        "--time_step",
        type=int,
        default=-1,
        help="Time step in seconds",
    )


## Reward

    parser.add_argument(
        "--alpha_temp",
        type=float,
        default=-1,
        help="Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )

    parser.add_argument(
        "--alpha_sig",
        type=float,
        default=-1,
        help="Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )

    parser.add_argument(
        "--temp_penalty_mode",
        type=str,
        default="config",
        help="Mode of temperature reward.",
    )

    parser.add_argument(
        "--alpha_ind_L2",
        type=float,
        default=-1,
        help="Coefficient of independant L2 in mixture temperature loss",
    )

    parser.add_argument(
        "--alpha_common_L2",
        type=float,
        default=-1,
        help="Coefficient of common L2 in mixture temperature loss")

    parser.add_argument(
        "--alpha_common_max",
        type=float,
        default=-1,
        help="Coefficient of common_max in mixture temperature loss") 

## Simulator


# Outdoors

    parser.add_argument(
        "--OD_temp_mode",
        type=str,
        default="config",
        help="Mode of outdoors temperature.",
    )

    parser.add_argument(
        "--no_solar_gain",
        action="store_true",
        help="Removes the solar gain from the simulation.",
    )


# House and HVAC
    parser.add_argument(
        "--cooling_capacity",
        type=int,
        default=-1,
        help="Default cooling capacity of the HVACs",
    )

    parser.add_argument(
        "--lockout_duration",
        type=int,
        default=-1,
        help="Default AC lockout duration, in seconds",
    )


# Noise
    parser.add_argument(
        "--house_noise_mode",
        type=str,
        default="config",
        help="Mode of noise over house parameters.",
    )

    parser.add_argument(
        "--house_noise_mode_test",
        type=str,
        default="train",
        help="Mode of noise over house parameters for test environment.",
    )

    parser.add_argument(
        "--hvac_noise_mode",
        type=str,
        default="config",
        help="Mode of noise over HVAC parameters.",
    )

    parser.add_argument(
        "--hvac_noise_mode_test",
        type=str,
        default="train",
        help="Mode of noise over HVAC parameters for test environment.",
    )


## Signal
    parser.add_argument(
        "--signal_mode",
        type=str,
        default="config",
        help="Mode of the noise on the power grid regulation signal simulation. Choices: [none, regular_steps, sinusoidals, config]",
    )

    parser.add_argument(
        "--base_power_mode",
        type=str,
        default="config",
        help="Mode for the base (low frequency) regulation signal simulation. Choices: [constant, interpolation, config]",
    )

    parser.add_argument(
        "--artificial_signal_ratio",
        type=float,
        default=1.0,
        help="Artificially multiply the base signal for experimental purposes.")

    parser.add_argument(
        "--artificial_signal_ratio_range",
        type=float,
        default=-1,
        help="Range from which the base signal is artificially multiplied or divided at every episode during training. Ex: 1 will not modify the signal. 3 will have signal modified between 1/3 and 3 times the base signal.")


## State

    parser.add_argument(
        "--state_day",
        choices = ['True','False'],
        default='True',
        help="Include day in the state")

    parser.add_argument(
        "--state_hour",
        choices = ['True','False'],
        default='True',
        help="Include hour in the state")

    parser.add_argument(
        "--state_solar_gain",
        choices = ['True','False'],
        default='False',
        help="Include solar gain in the state")

    parser.add_argument(
        "--state_thermal",
        choices = ['True','False'],
        default = 'True',
        help="Include outdoors temperature, and house thermal parameters, in the state.")



### Agent

    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        help="Type of agent (dqn, ppo)",
    )

## Agent communication constraints

    parser.add_argument(
        "--nb_agents_comm",
        type=int,
        default=-1,
        help="Maximal number of agents each agent can communicate with.",
    )

    parser.add_argument(
        "--agents_comm_mode",
        type=str,
        default="config",
        help="Mode for choosing the agents to communicate with. Can be 'neighbours' or 'random'",
    )



## PPO agent

# NN architecture

    parser.add_argument(
        "--layers_critic",
        type=str,
        default="config",
        help="List containing the number of neurons on each layers of the critic NN model",
    )

    parser.add_argument(
        "--layers_actor",
        type=str,
        default="config",
        help="List containing the number of neurons on each layers of the critic NN model",
    )

    parser.add_argument(
        "--layers_both",
        type=str,
        default="config",
        help="List containing the number of neurons on each layers of the critic NN model"
    )

# NN initialization

    parser.add_argument(
        "--net_seed",
        type=int,
        default=1,
        help="Neural network seed",
    )

# NN optimization

    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="Batch size",
    )

    parser.add_argument(
        "--lr_critic",
        type=float,
        default=-1,
        help="Learning rate of critic network"
        )

    parser.add_argument(
        "--lr_actor",
        type=float,
        default=-1,
        help="Learning rate of actor network"
        )

    parser.add_argument(
        "--lr_both",
        type=float,
        default=-1,
        help="Equal learning rate of actor and critic networks")

# RL optimization

    parser.add_argument(
        "--gamma",
        type=float,
        default=-1,
        help="Reward discount parameter"
        )


    parser.add_argument(
        "--clip_param",
        type=float,
        default=-1,
        help="PPO loss clipping parameter"
        )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=-1,
        help="PPO gradient norm maximum"
        )

    parser.add_argument(
        "--ppo_update_time",
        type=int,
        default=-1,
        help="Number of update rounds at each epoch"
        )    

    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=-1,
        help="Replay buffer capacity"
        )
# Training parameters

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

    opt = parser.parse_args()

    return opt


def cli_deploy(agents_dict):
    parser = argparse.ArgumentParser(description="Deployment options")

    parser.add_argument(
        "--base_power_mode",
        type=str,
        default="config",
        help="Mode for the base (low frequency) regulation signal simulation. Choices: [constant, interpolation, config]",
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=agents_dict.keys(),
        required=True,
        help="Agent for control",
    )

    parser.add_argument(
        "--render_after",
        type=int,
        default=-1,
        help="Delay in time steps before rendering",
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
        "--OD_temp_mode",
        type=str,
        default="config",
        help="Mode of outdoors temperature.",
    )

    parser.add_argument(
        "--no_solar_gain",
        action="store_true",
        help="Removes the solar gain from the simulation.",
    )

    parser.add_argument(
        "--nb_agents_comm",
        type=int,
        default=-1,
        help="Maximal number of agents each agent can communicate with.",
    )

    parser.add_argument(
        "--agents_comm_mode",
        type=str,
        default="config",
        help="Mode for choosing the agents to communicate with. Can be 'neighbours' or 'random'",
    )

    parser.add_argument(
        "--layers_critic",
        type=str,
        default="config",
        help="List containing the number of neurons on each layers of the critic NN model"
    )

    parser.add_argument(
        "--layers_actor",
        type=str,
        default="config",
        help="List containing the number of neurons on each layers of the critic NN model"
    )


    parser.add_argument(
        "--layers_both",
        type=str,
        default="config",
        help="List containing the number of neurons on each layers of the critic NN model"
    )

    
    parser.add_argument(
        "--start_stats_from",
        type=int,
        default=0,
        help="Number of time steps from which the stats are calculated.",
    )

    parser.add_argument(
        "--MPC_rolling_horizon",
        type=int,
        default=-1,
        help="Duration of the MPC rooling horizon in time step",
    )
    parser.add_argument(
        "--state_day",
        default='True',
        choices = ['True','False'],
        help="Include day in the state")

    parser.add_argument(
        "--state_hour",
        default='True',
        choices = ['True','False'],
        help="Include hour in the state")

    parser.add_argument(
        "--state_solar_gain",
        default='False',
        choices = ['True','False'],
        help="Include solar gain in the state")

    parser.add_argument(
        "--start_datetime_mode",
        default='config',
        help="Decide if start date time is 'fixed' or uniformly 'random'.")

    parser.add_argument(
        "--state_thermal",
        choices = ['True','False'],
        default = 'True',
        help="Include outdoors temperature, and house thermal parameters, in the state.")

    parser.add_argument(
        "--artificial_signal_ratio",
        type=float,
        default=1.0,
        help="Artificially multiply the base signal for experimental purposes."
        )


    opt = parser.parse_args()

    return opt

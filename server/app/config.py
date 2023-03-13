from pydantic import BaseModel, Extra, Field
import typing as t
import datetime


class HvacNoiseProperties(BaseModel):
    std_latent_cooling_fraction: float = Field(
        0.05,
        description="Standard deviation of the latent cooling fraction of the HVAC.",
    )
    factor_COP_low: float = Field(
        0.95,
        description="Lowest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_COP_high: float = Field(
        1.05,
        description="Highest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_cooling_capacity_low: float = Field(
        0.9,
        description="Lowest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    factor_cooling_capacity_high: float = Field(
        1.1,
        description="Highest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    lockout_noise: int = Field(
        0,
        description="Lockout noise to add to the lockout duration of the HVAC.",  # TODO check if this is correct
    )
    cooling_capacity_list: t.List[int] = Field(
        [12500, 15000, 17500],
        description="List of cooling capacities to choose from randomly.",  # TODO check if this is correct
    )


class HvacProperties(
    BaseModel,
    extra=Extra.forbid,
    allow_mutation=False,
    validate_assignment=True,
    freeze=True,
):
    cop: float = Field(
        2.5,
        description="coefficient of performance (ratio between cooling capacity and electric power consumption).",
        gt=0,
    )
    cooling_capacity: int = Field(
        15000,
        description='Rate of "negative" heat transfer produced by the HVAC (W).',
        gt=0,
    )
    latent_cooling_fraction: float = Field(
        0.35,
        description="Float between 0 and 1, fraction of sensible cooling (temperature) which is latent cooling (humidity).",
        gt=0,
        lt=1,
    )
    lockout_duration: int = Field(
        40,
        description="Duration of lockout (hardware constraint preventing to turn on the HVAC for some time after turning off), in seconds",
    )

    @property
    def max_consumption(self) -> float:
        return self.cooling_capacity / self.cop


class BuildingNoiseProperties(BaseModel):
    std_start_temp: float = Field(
        3.0,
        description="Standard deviation of the initial temperature of the house (Celsius).",
    )
    std_target_temp: float = Field(
        1.0,
        description="Standard deviation of the target temperature of the house (Celsius).",
    )
    factor_thermo_low: float = Field(
        0.9,
        description="Factor to multiply the standard deviation of the target temperature of the house (Celsius).",
    )
    factor_thermo_high: float = Field(
        1.1,
        description="Factor to multiply the standard deviation of the target temperature of the house (Celsius).",
    )


class ThermalProperties(BaseModel):
    Ua: float = Field(
        2.18e02,
        description="House walls conductance (W/K). Multiplied by 3 to account for drafts (according to https://dothemath.ucsd.edu/2012/11/this-thermal-house/)",
    )
    Ca: float = Field(
        9.08e05,
        description="Air thermal mass in the house (J/K): 3 * (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)",
    )
    Hm: float = Field(
        2.84e03,
        description="House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.14 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))",
    )
    Cm: float = Field(
        3.45e06,
        description="House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)",
    )


class BuildingProperties(ThermalProperties):
    target_temp: float = Field(
        20,
        description="Desired temperature in the house (Celsius).",
    )
    deadband: float = Field(
        0,
        description="Deadband around the target temperature (Celsius).",
    )
    init_air_temp: float = Field(
        20,
        description="Initial temperature of the air in the house (Celsius).",
    )
    init_mass_temp: float = Field(
        20,
        description="Initial temperature of the mass in the house (Celsius).",
    )
    solar_gain: bool = Field(
        True,
        description="Whether to include solar gain in the simulation.",
    )
    nb_hvacs: int = Field(
        1,
        description="Number of HVACs in the house.",
    )
    window_area: float = Field(
        7.175,
        description="Gross window area (m2).",
    )
    shading_coeff: float = Field(
        0.67,
        description="Window Solar Heat Gain Coefficient, look-up table in Gridlab reference",
    )


class PPOProp(BaseModel):
    """Properties for PPO agent."""

    actor_layers: list[int] = Field(
        [100, 100],
        description="List of layer sizes for the actor network.",
    )
    critic_layers: list[int] = Field(
        [100, 100],
        description="List of layer sizes for the critic network.",
    )
    gamma: float = Field(
        0.99,
        description="Discount factor for the reward.",
    )
    lr_critic: float = Field(
        3e-3,
        description="Learning rate for the critic network.",
    )
    lr_actor: float = Field(
        3e-3,
        description="Learning rate for the actor network.",
    )
    clip_param: float = Field(
        0.2,
        description="Clipping parameter for the PPO loss.",
    )
    max_grad_norm: float = Field(
        0.5,
        description="Maximum norm for the gradient clipping.",
    )
    ppo_update_time: int = Field(
        10,
        description="Update time for the PPO agent.",
    )
    batch_size: int = Field(
        256,
        description="Batch size for the PPO agent.",
    )
    zero_eoepisode_return: bool = Field(
        False,
        # description="Whether to zero the episode return when the episode ends.",
    )


class MAPPOProp(PPOProp):
    """Properties for MAPPO agent."""


class DDPGProp(BaseModel):
    """Properties for MAPPO agent."""

    actor_hidden_dim: int = Field(
        256,
        description="Hidden dimension for the actor network.",
    )
    critic_hidden_dim: int = Field(
        256,
        description="Hidden dimension for the critic network.",
    )
    lr_critic: float = Field(
        3e-3,
        description="Learning rate for the critic network.",
    )
    lr_actor: float = Field(
        3e-3,
        description="Learning rate for the actor network.",
    )
    soft_tau: float = Field(
        0.01,
        description="Soft target update parameter.",
    )
    clip_param: float = Field(
        0.2,
        description="Clipping parameter for the PPO loss.",
    )
    max_grad_norm: float = Field(
        0.5,
        description="Maximum norm for the gradient clipping.",
    )
    ddpg_update_time: int = Field(
        10,
        description="Update time for the DDPG agent.",
    )
    batch_size: int = Field(
        64,
        description="Batch size for the DDPG agent.",
    )
    buffer_capacity: int = Field(
        524288,
        description="Capacity of the replay buffer.",
    )
    episode_num: int = Field(
        10000,
        # description="Number of episodes for the MAPPO agent.",
    )
    learn_interval: int = Field(
        100,
        description="Learning interval for the MAPPO agent.",
    )
    random_steps: int = Field(
        100,
        # description="Number of random steps for the MAPPO agent.",
    )
    gumbel_softmax_tau: float = Field(
        1.0,
        description="Temperature for the gumbel softmax distribution.",
    )
    DDPG_shared: bool = Field(
        True,
        # description="Whether to use the shared DDPG network.",
    )


class DQNProp(BaseModel):
    """Properties for DQN agent."""

    network_layers: list[int] = Field(
        [100, 100],
        description="List of layer sizes for the DQN network.",
    )
    gamma: float = Field(
        0.99,
        description="Discount factor for the reward.",
    )
    tau: float = Field(
        0.001,
        description="Soft target update parameter.",
    )
    lr: float = Field(
        3e-3,
        description="Learning rate for the DQN network.",
    )
    buffer_capacity: int = Field(
        524288,
        description="Capacity of the replay buffer.",
    )
    batch_size: int = Field(
        256,
        description="Batch size for the DQN agent.",
    )
    epsilon_decay: float = Field(
        0.99998,
        description="Epsilon decay rate for the DQN agent.",
    )
    min_epsilon: float = Field(
        0.01,
        description="Minimum epsilon for the DQN agent.",
    )


class MPCProp(BaseModel):
    """Properties for MPC agent."""

    rolling_horizon: int = Field(
        15,
        description="Rolling horizon for the MPC agent.",
    )


class CLIConfig(BaseModel):
    """Properties ported from the CIL calls."""

    experiment_name: str = Field(
        "default",
        description="Name of the experiment.",
    )
    wandb: bool = Field(
        False,
        description="Whether to use wandb.",
    )
    log_metrics_path: str = Field(
        "",
        description="Path to the metrics file.",
    )
    nb_time_steps: int = Field(
        100000,
        description="Number of time steps for the experiment.",
    )
    save_actor_name: str = Field(
        "",
        description="Name of the actor to save.",
    )
    nb_inter_saving_actor: int = Field(
        0,
        description="Number of time steps between saving the actor.",
    )


class PenaltyProperties(BaseModel):
    mode: str = "common_L2"
    alpha_ind_l2: float = 1.0
    alpha_common_l2: float = 1.0
    alpha_common_max: float = 0.0


class RewardProperties(BaseModel):
    """Properties of the reward function."""

    alpha_temp: float = Field(
        1.0,
        description="Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    alpha_sig: float = Field(
        1.0,
        description="Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    norm_reg_signal: int = Field(
        7500,
        description="Average power use, for signal normalization.",
    )
    penalty_props: PenaltyProperties = Field(PenaltyProperties())


class StateProperties(BaseModel):
    """Properties of the state space."""

    hour: bool = Field(
        False,
        description="Whether to include the hour of the day in the state space.",
    )
    day: bool = Field(
        False,
        description="Whether to include the day of the week in the state space.",
    )
    solar_gain: bool = Field(
        False,
        description="Whether to include solar gain in the state space.",
    )
    thermal: bool = Field(
        False,
        description="Whether to include thermal state in the state space.",
    )
    hvac: bool = Field(
        False,
        description="Whether to include hvac state in the state space.",
    )


class EnvironmentProperties(BaseModel):
    """Properties of the environment."""

    start_datetime: datetime.datetime = Field(
        datetime.datetime(2021, 1, 1, 0, 0, 0),
        description="Start date and time (Y-m-d H:M:S).",
    )
    start_datetime_mode: t.Union[
        t.Literal["individual_L2"],
        t.Literal["common_L2"],
        t.Literal["common_max"],
        t.Literal["mixture"],
    ] = Field(
        "fixed",
        description="Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)",
    )
    time_step: datetime.timedelta = Field(datetime.timedelta(0, 4))  # Time step (H:M:S)
    state_properties: StateProperties
    message_properties: t.Dict[str, bool] = {"thermal": False, "hvac": False}
    reward_properties: RewardProperties


class MarlConfig(BaseModel, extra=Extra.forbid):
    """Configuration for MARL environment."""

    default_house_prop: BuildingProperties
    noise_house_prop: BuildingNoiseProperties
    default_hvac_prop: HvacProperties
    noise_hvac_prop: HvacNoiseProperties
    default_env_prop: EnvironmentProperties
    PPO_prop: PPOProp
    MAPPO_prop: MAPPOProp
    DDPG_prop: DDPGProp
    DQN_prop: DQNProp
    MPC_prop: MPCProp

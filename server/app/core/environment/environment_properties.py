import datetime
from typing import List, Literal, TypedDict, Union

from pydantic import BaseModel, Field

from app.core.environment.cluster.cluster_properties import (
    AgentsCommunicationProperties,
    TemperatureProperties,
)
from app.core.environment.power_grid.power_grid_properties import (
    PowerGridProperties,
)


class HvacNoiseProperties(BaseModel):
    """Noise applied to the HVAC properties."""

    std_latent_cooling_fraction: float = Field(
        default=0.05,
        description="Standard deviation of the latent cooling fraction of the HVAC.",
    )
    factor_COP_low: float = Field(
        default=0.95,
        description="Lowest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_COP_high: float = Field(
        default=1.05,
        description="Highest random factor for COP to multiply the coefficient of performance of the HVAC.",
    )
    factor_cooling_capacity_low: float = Field(
        default=0.9,
        description="Lowest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    factor_cooling_capacity_high: float = Field(
        default=1.1,
        description="Highest random factor for cooling_capacity to multiply the cooling capacity of the HVAC.",
    )
    lockout_noise: int = Field(
        default=0,
        description="Lockout noise to add to the lockout duration of the HVAC.",  # TODO check if this is correct
    )
    cooling_capacity_list: List[int] = Field(
        default=[12500, 15000, 17500],
        description="List of cooling capacities to choose from randomly.",  # TODO check if this is correct
    )


class HvacProperties(BaseModel):
    """Model describing the properties of the HVAC."""

    cop: float = Field(
        default=2.5,
        description="coefficient of performance (ratio between cooling capacity and electric power consumption).",
        gt=0,
    )
    cooling_capacity: float = Field(
        default=15000.0,
        description='Rate of "negative" heat transfer produced by the HVAC (W).',
        gt=0,
    )
    latent_cooling_fraction: float = Field(
        default=0.35,
        description="Float between 0 and 1, fraction of sensible cooling (temperature) which is latent cooling (humidity).",
        gt=0,
        lt=1,
    )
    lockout_duration: int = Field(
        default=40,
        description="Duration of lockout (hardware constraint preventing to turn on the HVAC for some time after turning off), in seconds",
    )
    noise_prop: HvacNoiseProperties = HvacNoiseProperties()

    @property
    def max_consumption(self) -> float:
        """Maximum consumption of the HVAC.

        (ratio of cooling capacity and coefficient of performance).
        """
        return self.cooling_capacity / self.cop


class BuildingNoiseProperties(BaseModel):
    std_start_temp: float = Field(
        default=3.0,
        description="Standard deviation of the initial temperature of the house (Celsius).",
    )
    std_target_temp: float = Field(
        default=1.0,
        description="Standard deviation of the target temperature of the house (Celsius).",
    )
    factor_thermo_low: float = Field(
        default=0.9,
        description="Factor to multiply the standard deviation of the target temperature of the house (Celsius).",
    )
    factor_thermo_high: float = Field(
        default=1.1,
        description="Factor to multiply the standard deviation of the target temperature of the house (Celsius).",
    )


class ThermalProperties(BaseModel):
    Ua: float = Field(
        default=2.18e02,
        description="House walls conductance (W/K). Multiplied by 3 to account for drafts (according to https://dothemath.ucsd.edu/2012/11/this-thermal-house/)",
    )
    Ca: float = Field(
        default=9.08e05,
        description="Air thermal mass in the house (J/K): 3 * (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)",
    )
    Hm: float = Field(
        default=2.84e03,
        description="House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.14 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))",
    )
    Cm: float = Field(
        default=3.45e06,
        description="House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)",
    )


class BuildingProperties(ThermalProperties):
    target_temp: float = Field(
        default=20.0,
        description="Desired temperature in the house (Celsius).",
    )
    deadband: float = Field(
        default=0.0,
        description="Deadband around the target temperature (Celsius).",
    )
    init_air_temp: float = Field(
        default=20.0,
        description="Initial temperature of the air in the house (Celsius).",
    )
    init_mass_temp: float = Field(
        default=20.0,
        description="Initial temperature of the mass in the house (Celsius).",
    )
    solar_gain: bool = Field(
        default=True,
        description="Whether to include solar gain in the simulation.",
    )
    window_area: float = Field(
        default=7.175,
        description="Gross window area (m2).",
    )
    shading_coeff: float = Field(
        default=0.67,
        description="Window Solar Heat Gain Coefficient, look-up table in Gridlab reference",
    )
    noise_prop: BuildingNoiseProperties = BuildingNoiseProperties()
    hvac_prop: HvacProperties = HvacProperties()


class PenaltyProperties(BaseModel):
    mode: Literal["common_L2", "individual_L2", "common_max_error", "mixture"] = Field(
        default="individual_L2", description="Mode of temperature penalty"
    )
    alpha_ind_l2: float = 1.0
    alpha_common_l2: float = 1.0
    alpha_common_max: float = 0.0


class RewardProperties(BaseModel):
    """Properties of the reward function."""

    alpha_temp: float = Field(
        default=1.0,
        description="Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    alpha_sig: float = Field(
        default=1.0,
        description="Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.",
    )
    norm_reg_sig: int = Field(
        default=7500,
        description="Average power use, for signal normalization.",
        alias="norm_reg_sig",
    )
    penalty_props: PenaltyProperties = PenaltyProperties()
    sig_penalty_mode: Literal["common_L2"] = Field(
        default="common_L2",
        description="Mode of signal penalty (only common_L2 available)",
    )


class StateProperties(BaseModel):
    """Properties of the state space."""

    hour: bool = Field(
        default=False,
        description="Whether to include the hour of the day in the state space.",
    )
    day: bool = Field(
        default=False,
        description="Whether to include the day of the week in the state space.",
    )
    solar_gain: bool = Field(
        default=False,
        description="Whether to include solar gain in the state space.",
    )
    thermal: bool = Field(
        default=False,
        description="Whether to include thermal state in the state space.",
    )
    hvac: bool = Field(
        default=False,
        description="Whether to include hvac state in the state space.",
    )


class MessageProperties(BaseModel):
    """Properties of the message space."""

    thermal: bool = Field(
        default=False,
        description="Whether to include thermal state in the message space.",
    )
    hvac: bool = Field(
        default=False,
        description="Whether to include hvac state in the message space.",
    )


class ClusterPropreties(BaseModel):
    nb_agents: int = Field(
        default=10,
        description="Number of agents in the cluster.",
    )
    nb_agents_comm: int = Field(
        default=10,
        description="Maximal number of houses a single house communicates with.",
    )
    # TODO: make field
    agents_comm_prop: AgentsCommunicationProperties = AgentsCommunicationProperties()
    message_prop: MessageProperties = MessageProperties()
    house_prop: BuildingProperties = BuildingProperties()


class EnvironmentProperties(BaseModel):
    """Properties of the environment."""

    start_datetime: datetime.datetime = Field(
        default=datetime.datetime(2021, 1, 1, 0, 0, 0),
        description="Start date and time (Y-m-d H:M:S).",
    )
    start_datetime_mode: Literal["fixed", "random"] = Field(
        default="random",
        description="Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)",
    )
    time_step: datetime.timedelta = Field(
        default=datetime.timedelta(0, 4),
        description="How long a timestep should take (in seconds).",
    )

    temp_prop: TemperatureProperties = TemperatureProperties()
    state_prop: StateProperties = StateProperties()
    reward_prop: RewardProperties = RewardProperties()
    cluster_prop: ClusterPropreties = ClusterPropreties()
    power_grid_prop: PowerGridProperties = PowerGridProperties()


class HvacMessage(TypedDict, total=False):
    """Structure of the hvac message dictionnary."""

    cop: float
    cooling_capacity: float
    latent_cooling_fraction: float
    seconds_since_off: int
    curr_consumption: float
    max_consumption: float
    lockout_duration: int


class BuildingMessage(HvacMessage, total=False):
    current_temp_diff_to_target: float
    Ua: float
    Ca: float
    Hm: float
    Cm: float


class EnvironmentObsDict(TypedDict, total=False):
    OD_temp: float
    datetime: datetime.datetime
    cluster_hvac_power: float
    cop: float
    cooling_capacity: float
    latent_cooling_fraction: float
    lockout_duration: int
    seconds_since_off: int
    turned_on: Union[bool, int]
    lockout: Union[bool, int]
    target_temp: float
    deadband: float
    Ua: float
    Ca: float
    Hm: float
    Cm: float
    indoor_temp: float
    mass_temp: float
    solar_gain: float
    message: Union[List[BuildingMessage], list]
    reg_signal: float

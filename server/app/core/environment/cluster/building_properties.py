from pydantic import BaseModel, Field


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
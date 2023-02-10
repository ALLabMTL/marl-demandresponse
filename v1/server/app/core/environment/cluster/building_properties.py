from pydantic import BaseModel


class BuildingNoiseProperties(BaseModel):
    std_start_temp: float = 3.0
    std_target_temp: float = 1.0
    factor_thermo_low: float = 0.9
    factor_thermo_high: float = 1.1

class ThermalProperties(BaseModel):
    Ua: float = 2.18e02
    Ca: float = 9.08e05
    Hm: float = 2.84e03
    Cm: float = 3.45e06

class BuildingProperties(BaseModel):
    Ua: float = 2.18e02
    Ca: float = 9.08e05
    Hm: float = 2.84e03
    Cm: float = 3.45e06
    target_temp: float = 20.0
    deadband: float = 0.0
    init_air_temp: float = 20.0
    init_mass_temp: float = 20.0
    solar_gain: bool = True
    nb_hvacs: int = 1
    window_area: float = 7.175
    shading_coeff: float = 0.67
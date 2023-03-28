from typing import List, Union

from pydantic import BaseModel


class SignalProperties(BaseModel):
    # TODO: implement other modes
    mode: str = "regular_steps"
    amplitude_ratios: Union[float, List[float]] = [0.1, 0.3]
    amplitude_per_hvac: int = 6000
    nb_octaves: int = 5
    octaves_step: int = 5
    period: int = 300
    periods: List[int] = [400, 1200]


class BasePowerProperties(BaseModel):
    # TODO: Think about a better way to do it
    mode: str = "constant"
    avg_power_per_hvac: int = 4200
    init_signal_per_hvac: int = 910
    path_datafile: str = "./monteCarlo/mergedGridSearchResultFinal.npy"
    path_parameter_dict: str = "./monteCarlo/interp_parameters_dict.json"
    path_dict_keys: str = "./monteCarlo/interp_dict_keys.csv"
    interp_update_period: int = 300
    interp_nb_agents: int = 100


class PowerGridProperties(BaseModel):
    artificial_signal_ratio_range: int = 1
    base_power_props: BasePowerProperties = BasePowerProperties()
    signal_properties: SignalProperties = SignalProperties()
    artificial_ratio: float = 1.0

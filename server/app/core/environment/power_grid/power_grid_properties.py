from typing import List

from pydantic import BaseModel


class SignalProperties(BaseModel):
    """
    Store the properties related to the power grid signal.

    Attributes:
        mode (str): The mode of the power grid signal (default: "perlin").
        amplitude_ratios (List[float]): The amplitude ratios of the power grid signal (default: [0.1, 0.3]).
        amplitude_per_hvac (int): The amplitude per HVAC of the power grid signal (default: 6000).
        nb_octaves (int): The number of octaves used to compute the Perlin noise (default: 5).
        octaves_step (int): The step between octaves used to compute the Perlin noise (default: 5).
        period (int): The period of the power grid signal (default: 300).
        periods (List[int]): The periods of the sinusoidal signals (default: [400, 1200]).
    """

    # TODO: implement other modes
    mode: str = "perlin"
    amplitude_ratios: List[float] = [0.1, 0.3]
    amplitude_per_hvac: int = 6000
    nb_octaves: int = 5
    octaves_step: int = 5
    period: int = 300
    periods: List[int] = [400, 1200]


class BasePowerProperties(BaseModel):
    """
    Store the properties related to the base power of the power grid.

    Attributes:
        mode (str): The mode of the base power (default: "constant").
        avg_power_per_hvac (int): The average power per HVAC (default: 4200).
        init_signal_per_hvac (int): The initial signal per HVAC (default: 910).
        path_datafile (str): The path to the data file (default: "./monteCarlo/mergedGridSearchResultFinal.npy").
        path_parameter_dict (str): The path to the parameter dictionary (default: "./monteCarlo/interp_parameters_dict.json").
        path_dict_keys (str): The path to the dictionary keys (default: "./monteCarlo/interp_dict_keys.csv").
        interp_update_period (int): The update period of the interpolator (default: 300).
        interp_nb_agents (int): The number of agents used to compute the interpolator (default: 100).
    """

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
    """
    Store the properties related to the power grid.

    Attributes:
        artificial_signal_ratio_range (int): The range of the artificial signal ratio (default: 1).
        base_power_props (BasePowerProperties): The properties related to the base power (default: BasePowerProperties()).
        signal_properties (SignalProperties): The properties related to the power grid signal (default: SignalProperties()).
        artificial_ratio (float): The artificial ratio of the power grid (default: 1.0).
    """

    artificial_signal_ratio_range: int = 1
    base_power_props: BasePowerProperties = BasePowerProperties()
    signal_properties: SignalProperties = SignalProperties()
    artificial_ratio: float = 1.0

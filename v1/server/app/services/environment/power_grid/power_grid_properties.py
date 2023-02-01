from pydantic import BaseModel


class SignalProperties(BaseModel):
    amplitude_ratios: float
    nb_octaves: int
    octaves_step: int
    period: int


class BasePowerProperties(BaseModel):
    constant: int


class PowerGridProperties(BaseModel):
    artificial_signal_ratio_range: int
    # Scale of artificial multiplicative factor randomly multiplied (or divided) at each episode
    # during training.
    # Ex: 1 will not modify signal. 3 will have signal between 33% and 300% of what is computed.
    base_power: BasePowerProperties()
    artificial_ratio: float
    signal_properties: SignalProperties()

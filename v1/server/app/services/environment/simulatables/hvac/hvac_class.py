from hvac_properties import HvacNoiseProperties, HvacProperties
from simulatable import Simulatable


class HVAC(Simulatable):
    initial_properties: HvacProperties()
    turned_on: bool
    seconds_since_off: float
    time_step: float

    def step() -> None:
        pass

    # computes the rate of heat transfer produced by the HVAC
    def get_Q(self):
        pass

    # computes the electric power consumption of the HVAC
    def power_consumption(self):
        pass

    def apply_noise(noise_properties: HvacNoiseProperties):
        pass

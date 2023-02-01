from typing import List

from v1.server.app.services.environment.cluster.buildings.building import Building
from v1.server.app.services.environment.cluster.buildings.house.house_properties import (
    HouseNoiseProperties,
    HouseProperties,
)
from v1.server.app.services.environment.simulatables.simulatable import Simulatable


class House(Building):
    house_properties: HouseProperties = HouseProperties()
    noise_properties: HouseNoiseProperties = HouseNoiseProperties()

    current_temperature: float
    current_mass_temperature: float
    simulated_objects: List[Simulatable]  # Pour l'instant on peut juste mettre hvac

    def __init__():
        pass

    def step() -> None:
        pass

    def update_temperature(self, od_temp, time_step) -> None:
        pass

    # def apply_noise(self, noise_properties: HouseNoiseProperties) -> None:
    #     noise_house_mode = noise_properties.noise_mode
    #     noise_house_params = noise_house_prop["noise_parameters"][noise_house_mode]

    #     # Gaussian noise: target temp
    #     self.properties.init_air_temp += abs(
    #         random.gauss(0, noise_properties.std_start_temp)
    #     )

    #     self.properties.init_mass_temp += abs(
    #         random.gauss(0, noise_properties.std_start_temp)
    #     )
    #     self.properties.target_temp += abs(
    #         random.gauss(0, noise_properties.std_target_temp)
    #     )

    #     # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance
    #     factor_Ua = random.triangular(
    #         noise_house_params["factor_thermo_low"],
    #         noise_house_params["factor_thermo_high"],
    #         1,
    #     )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    #     house_prop["Ua"] *= factor_Ua

    #     factor_Cm = random.triangular(
    #         noise_house_params["factor_thermo_low"],
    #         noise_house_params["factor_thermo_high"],
    #         1,
    #     )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    #     house_prop["Cm"] *= factor_Cm

    #     factor_Ca = random.triangular(
    #         noise_house_params["factor_thermo_low"],
    #         noise_house_params["factor_thermo_high"],
    #         1,
    #     )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    #     house_prop["Ca"] *= factor_Ca

    #     factor_Hm = random.triangular(
    #         noise_house_params["factor_thermo_low"],
    #         noise_house_params["factor_thermo_high"],
    #         1,
    #     )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    #     house_prop["Hm"] *= factor_Hm

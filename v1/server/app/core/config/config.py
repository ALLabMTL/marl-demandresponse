from pydantic import BaseModel, Extra
from app.core.environment.cluster.building_properties import BuildingNoiseProperties
from app.core.environment.cluster.hvac_properties import HvacNoiseProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.core.environment.cluster.hvac_properties import HvacProperties
from app.core.environment.cluster.building_properties import BuildingProperties
from app.config import config_dict


class NoiseHouseProp(BaseModel):
    noise_mode: str = "small_noise"
    noise_parameters: BuildingNoiseProperties


class NoiseHvacProp(BaseModel):
    noise_mode: str = "small_noise"
    noise_parameters: HvacNoiseProperties


class MarlConfig(BaseModel, extra=Extra.forbid):
    """Configuration for MARL environment"""

    default_house_prop: BuildingProperties
    noise_house_prop: NoiseHouseProp
    noise_house_prop_test: NoiseHouseProp
    default_hvac_prop: HvacProperties
    noise_hvac_prop: NoiseHvacProp
    noise_hvac_prop_test: NoiseHvacProp
    default_env_prop: EnvironmentProperties
    PPO_prop: dict
    MAPPO_prop: dict
    DDPG_prop: dict
    DQN_prop: dict
    MPC_prop: dict


print(config_dict)
my_config = MarlConfig(**config_dict)
print(my_config.json(indent=4))

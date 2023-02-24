from pydantic import BaseModel, Extra, validator, Field
from app.core.environment.cluster.building_properties import BuildingNoiseProperties
from app.core.environment.cluster.hvac_properties import HvacNoiseProperties
from app.core.environment.environment_properties import EnvironmentProperties
from app.core.environment.cluster.hvac_properties import HvacProperties
from app.core.environment.cluster.building_properties import BuildingProperties
from app.config import config_dict

from typing import Union


class Contained(BaseModel):
    nums: list[int]




class Container(BaseModel):
    noise_param: Contained
    noise_mode: str
    noise_resolved : Contained

    @property
    def noise_resolved(self):
        return self.noise_param

    get_noise_

    # def __init__(self, **data) -> None:
    #     if "contained" in data:
    #         ctd: Union[Contained, str] = data["contained"]
    #         templates = {
    #             "big_noise": Contained(nums=[1, 2, 3]),
    #             "small_noise": Contained(nums=[4, 5, 6]),
    #         }
    #         if isinstance(ctd, str):
    #             data["contained"] = templates[ctd]
    #         else:
    #             data["contained"] = ctd
    #     super().__init__(**data)


ctr = Container(contained="big_noise")
print(ctr.json(indent=4))
ctr = Container(contained="small_noise")
print(ctr.json(indent=4))
import json

# ctr = Container(**json.loads('{"contained": "big_noise"}'))
# print(ctr.json(indent=4))
# # ctr = Container(contained="sdklhghklsdglhk")
# # print(ctr.json(indent=4))

# print(Container.schema_json(indent=4)) 


import typing as t
class NoiseHouseProp(BaseModel):
    noise_mode: str = "small_noise"
    noise_parameters: t.Dict[str, BuildingNoiseProperties]


class NoiseHvacProp(BaseModel):
    noise_mode: str = "small_noise"
    noise_parameters: HvacNoiseProperties


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
    pass

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


class MarlConfig(BaseModel, extra=Extra.forbid):
    """Configuration for MARL environment."""

    default_house_prop: BuildingProperties
    noise_house_prop: NoiseHouseProp
    noise_house_prop_test: NoiseHouseProp
    default_hvac_prop: HvacProperties
    noise_hvac_prop: NoiseHvacProp
    noise_hvac_prop_test: NoiseHvacProp
    default_env_prop: EnvironmentProperties
    PPO_prop: PPOProp
    MAPPO_prop: MAPPOProp
    DDPG_prop: DDPGProp
    DQN_prop: DQNProp
    MPC_prop: MPCProp

if __name__ == "__main__":
    # TODO: Name config and config module name conflict
    # TODO: remove script part here, config should be at runtime
    print(config_dict)
    my_config = MarlConfig(**config_dict)
    with open("schema.json", "wb") as sche_file:
        sche = my_config.schema_json(indent=4)
        sche_file.write(sche.encode("utf-8"))
    print(my_config.json(indent=4))

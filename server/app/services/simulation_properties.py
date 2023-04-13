from typing import Literal

import pydantic


class SimulationProperties(pydantic.BaseModel):
    mode: Literal["train", "simulation"] = pydantic.Field(
        default="simulation",
        description="Mode of the simulation.",
    )
    agent: str = pydantic.Field(default="BangBang")
    nb_episodes: int = pydantic.Field(default=3, ge=1)
    nb_time_steps_test: int = pydantic.Field(default=1000, ge=1)
    nb_test_logs: int = pydantic.Field(default=100, ge=1)
    nb_logs: int = pydantic.Field(default=100, ge=1)
    nb_epochs: int = pydantic.Field(default=20, ge=1)
    net_seed: int = pydantic.Field(default=4, ge=0)
    start_stats_from: int = pydantic.Field(default=0, ge=0)
    log_metrics_path: str = pydantic.Field(
        default="",
        description="Path to the metrics file.",
    )
    nb_time_steps: int = pydantic.Field(
        default=100000,
        description="Number of time steps for the experiment.",
    )
    save_actor_name: str = pydantic.Field(
        default="",
        description="Name of the actor to save.",
    )
    nb_inter_saving_actor: int = pydantic.Field(
        default=1, description="Number of time steps between saving the actor.", ge=1
    )

import pydantic


class ControllerPropreties(pydantic.BaseModel):
    nb_episodes: int = pydantic.Field(default=3, ge=1)
    nb_time_steps: int = pydantic.Field(default=1000, ge=1)
    nb_test_logs: int = pydantic.Field(default=100, ge=1)
    nb_logs: int = pydantic.Field(default=100, ge=1)
    actor_name: str = pydantic.Field(default="GreedyMyopic")
    net_seed: int = pydantic.Field(default=4, ge=0)
    agent: str = pydantic.Field(default="GreedyMyopic")
    start_stats_from: int = pydantic.Field(default=0, ge=0)

import pydantic


class TemperatureProperties(pydantic.BaseModel):
    """
    Pydantic model for temperature properties.
    """

    day_temp: float = pydantic.Field(
        default=26.0,
        description="Day temperature.",
    )
    night_temp: float = pydantic.Field(
        default=20.0,
        description="Night temperature.",
    )
    temp_std: float = pydantic.Field(
        default=1.0,
        description="Standard deviation of the temperature.",
    )
    random_phase_offset: bool = pydantic.Field(
        default=False,
        description="Whether to add a random phase offset to the temperature.",
    )
    phase: float = pydantic.Field(
        default=0.0,
        description="Phase offset of the temperature.",
    )


class AgentsCommunicationProperties(pydantic.BaseModel):
    """
    Pydantic model for agent communication properties.
    """

    mode: str = "neighbours"
    row_size: int = 5
    max_communication_distance: int = 2
    max_nb_agents_communication: int = 10


class MessageProperties(pydantic.BaseModel):
    """
    Pydantic model for message properties.
    """

    thermal: bool = False
    hvac: bool = False

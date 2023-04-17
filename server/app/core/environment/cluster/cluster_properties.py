from pydantic import BaseModel


class TemperatureProperties(BaseModel):
    mode: str = "noisy_sinusoidal"
    day_temp: float = 30.0
    night_temp: float = 23.0
    temp_std_deviation: float = 0.5
    random_phase_offset: bool = False
    phase: float = 0.0


class AgentsCommunicationProperties(BaseModel):
    mode: str = "neighbours"
    row_size: int = 5
    max_communication_distance: int = 2
    max_nb_agents_communication: int = 10


class MessageProperties(BaseModel):
    thermal: bool = False
    hvac: bool = False

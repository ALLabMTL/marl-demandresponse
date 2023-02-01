from pydantic import BaseModel
from v1.server.app.utils.utils import get_templates

AGENTS_COMMUNICATION_TEMPLATES_FILE_PATH = "v1/server/app/core/env/buildings/cluster_building/agents_communication_template.json"
TEMPERATURE_TEMPLATES_FILE_PATH = (
    "v1/server/app/core/env/buildings/cluster_building/temperature_templates.json"
)

agents_communication_templates = get_templates(AGENTS_COMMUNICATION_TEMPLATES_FILE_PATH)
communication_modes = agents_communication_templates.keys()
temperature_mode_templates = get_templates(TEMPERATURE_TEMPLATES_FILE_PATH)
temperature_modes = temperature_mode_templates.keys()


class TemperatureProperties(BaseModel):
    day_temperature: float
    night_temperature: float
    temperature_standard_deviation: float
    random_phase_offset: bool


class AgentsCommunicationProperties(BaseModel):
    row_size: int = 5
    max_communication_distance: int = 2


class ClusterProperties(BaseModel):
    nb_agents: int  # Number of houses
    max_nb_agents_communication: int  # Maximal number of houses a single house communicates with
    # TODO: faire un validator avec les keys du template
    temperature_mode: str
    agents_communication_mode: str
    # agents_communication_properties: AgentsCommunicationProperties()

from datetime import datetime, timedelta
from typing import Dict, List, Union

from app.core.environment.cluster.building import Building
from app.core.environment.cluster.cluster_properties import (
    AgentsCommunicationProperties,
    MessageProperties,
)
from app.core.environment.simulatable import Simulatable
from app.core.environment.cluster.agent_communication_builder import (
    AgentCommunicationBuilder,
)


class Cluster(Simulatable):
    # TODO: maybe we should put them in same model (static properties)
    agents_comm_properties: AgentsCommunicationProperties
    message_properties: MessageProperties
    nb_agents: int
    buildings: List[Building]
    current_power: int
    max_power: int
    agent_communicators: Dict[int, List[int]]
    nb_hvacs: int
    current_power_consumption: int

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> dict:
        # TODO: Initialize values with parser service
        self.agents_comm_properties = AgentsCommunicationProperties()
        self.message_properties = MessageProperties()
        self.nb_agents = 10
        self.buildings = [Building() for _ in range(self.nb_agents)]
        self.max_power = 0
        self.nb_hvacs = 0
        self.current_power_consumption = 0
        for building in self.buildings:
            # TODO: put power_consumptions in building class
            self.current_power_consumption += building.get_power_consumption()
            self.max_power += building.max_consumption
            self.nb_hvacs += building.initial_properties.nb_hvacs
        self.communication_builder = AgentCommunicationBuilder(
            agents_comm_props=self.agents_comm_properties, nb_agents=self.nb_hvacs
        )
        self.agent_communicators = self.communication_builder.get_comm_link_list()

        return self._get_obs()

    def _step(
        self,
        od_temp: float,
        action_dict: Dict[int, bool],
        date_time: datetime,
        time_step: timedelta,
    ) -> dict:
        self.current_power_consumption = 0
        for building_id, building in enumerate(self.buildings):
            if building_id in action_dict.keys():
                command = action_dict[building_id]
            else:
                command = False
            building._step(od_temp, time_step, date_time, command)
            self.current_power_consumption += building.get_power_consumption()
        return self._get_obs()

    def message(self, building_id: int) -> List[Dict[str, Union[int, float]]]:
        if self.agents_comm_properties.mode == "random_sample":
            id_houses_messages = self.communication_builder.get_random_sample(
                building_id
            )
        else:
            id_houses_messages = self.agent_communicators[building_id]

        message = []
        for id_house_message in id_houses_messages:
            message.append(
                self.buildings[id_house_message].message(
                    self.message_properties.thermal,
                    self.message_properties.hvac,
                )
            )
        return message

    def _get_obs(self) -> Dict[int, dict]:
        state_dict = {}
        # TODO: we need to change this into models and maybe
        # move this to parser service
        for building_id, building in enumerate(self.buildings):
            for hvac_id in range(self.nb_hvacs):
                state_dict.update({building_id + hvac_id: building._get_obs()})
                state_dict[building_id + hvac_id].update(
                    {
                        "cluster_hvac_power": self.current_power_consumption,
                        "message": self.message(hvac_id),
                    }
                )
        return state_dict

    def apply_noise(self) -> None:
        for building in self.buildings:
            building.apply_noise()

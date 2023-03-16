from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from app.core.environment.cluster.building import Building, BuildingObsDict
from app.core.environment.cluster.cluster_properties import (
    AgentsCommunicationProperties,
    MessageProperties,
)
from app.core.environment.environment_properties import ClusterPropreties
from app.core.environment.simulatable import Simulatable
from app.utils.logger import logger


class ClusterObsDict(BuildingObsDict):
    """Observation dictionary for Cluster."""

    cluster_hvac_power: float
    message: List[dict]


class Cluster(Simulatable):
    # TODO: maybe we should put them in same model (static properties)
    agents_comm_properties: AgentsCommunicationProperties
    message_properties: MessageProperties
    buildings: List[Building]
    agent_communicators: Dict[int, List[int]]

    @property
    def current_power(self) -> int:
        return sum([building.get_power_consumption() for building in self.buildings])

    @property
    def max_power(self) -> int:
        return sum([building.max_consumption for building in self.buildings])

    @property
    def nb_hvacs(self) -> int:
        return sum(
            [building.initial_properties.nb_hvacs for building in self.buildings]
        )

    def __init__(self, cluster_prop: ClusterPropreties) -> None:
        self.nb_agents = 10
        self._reset()

    def _reset(self) -> dict:
        # TODO: Initialize values with parser service
        self.agents_comm_properties = AgentsCommunicationProperties()
        self.message_properties = MessageProperties()
        self.buildings = [Building() for _ in range(self.nb_agents)]
        self.agent_communicators = {}
        self.current_power_consumption = 0
        self.build_agents_comm_links()

    def _step(
        self,
        od_temp: float,
        action_dict: Dict[int, bool],
        date_time: datetime,
        time_step: timedelta,
    ) -> None:
        self.current_power_consumption = 0

        # TODO: we need to change this if we are doing multiple hvacs
        for building_id, building in enumerate(self.buildings):
            if building_id in action_dict.keys():
                command = action_dict[building_id]
            else:
                logger.warn("")
                command = False
            building._step(od_temp, time_step, date_time, command)
            self.current_power_consumption += building.get_power_consumption()

    def message(self, thermal: bool, hvac: bool) -> dict:
        # TODO: we need to implement this method
        # to implement this method
        # for each building in the cluster, get state

        # for building in self.buildings:
        #     ids_building_messages = self.agent_communicators[index(building)]

        #     agent_ids = building.get_state()
        #     self.current_state["message"]= []
        #     for id_house_message in ids_houses_messages:
        #         self.current_state["message"].append(
        #             self.buildings[id_house_message].message(
        #                 self.env_prop["message_properties"]
        #             )
        #         )
        pass

    def _get_obs(self) -> Dict[int, Dict]:
        state_dict = {}
        # TODO: we need to change this into models and maybe
        # move this to parser service
        for building_id, building in enumerate(self.buildings):
            for hvac_id, _ in enumerate(building.hvacs):
                state_dict.update({building_id + hvac_id: building._get_obs().dict()})
                state_dict[building_id + hvac_id].update(
                    {"cluster_hvac_power": self.current_power_consumption}
                )
                id_houses_messages = self.agent_communicators[building_id]

                state_dict[building_id]["message"] = []
                for id_house_message in id_houses_messages:
                    state_dict[building_id]["message"].append(
                        self.buildings[id_house_message].message(True, False)
                    )
        return {k: ClusterObsDict(**v).dict() for k, v in state_dict.items()}

    def build_agents_comm_links(self) -> None:
        nb_comm = np.minimum(
            self.agents_comm_properties.max_nb_agents_communication,
            (len(self.buildings) - 1),
        )
        # This is to get the neighbours of each agent in a circular fashion,
        # if agent_id is 5, the half before will be [0, 1, 2, 3, 4] and half after will be [6, 7, 8, 9, 10]
        # if agent_id is 1, the half before will be [7, 8, 9, 10, 0] and half after will be [2, 3, 4, 5, 6]
        # TODO: implement other communication links than neighboors and make it prettier
        agent_ids = [i for i, _ in enumerate(self.buildings)]
        for agent_id in agent_ids:
            half_before = [
                (agent_id - int(np.floor(nb_comm / 2)) + i) % len(agent_ids)
                for i in range(int(np.floor(nb_comm / 2)))
            ]
            half_after = [
                (agent_id + 1 + i) % len(agent_ids)
                for i in range(int(np.ceil(nb_comm / 2)))
            ]
            self.agent_communicators[agent_id] = half_before + half_after
        # if self.agents_communication_properties.mode == "random_sample":
        #     possible_ids = deepcopy(agent_ids)
        #     nb_comm = np.minimum(
        #         self.agents_communication_properties.max_nb_agents_communication,
        #         self.agents_communication_properties.nb_agents - 1,
        #     )
        #     possible_ids.remove(house_id)
        #     ids_houses_messages = random.sample(possible_ids, k=nb_comm)

        # else:
        #     ids_houses_messages = self.agent_communicators[house_id]

    def apply_noise(self) -> None:
        for building in self.buildings:
            building.apply_noise()

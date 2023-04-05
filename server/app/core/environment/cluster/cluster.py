from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List


from app.core.environment.cluster.building import Building
from app.core.environment.simulatable import Simulatable
from app.core.environment.cluster.agent_communication_builder import (
    AgentCommunicationBuilder,
)

from app.core.environment.environment_properties import (
    ClusterPropreties,
    EnvironmentObsDict,
    BuildingMessage,
)


class Cluster(Simulatable):
    init_props: ClusterPropreties
    current_power_consumption: float
    max_power: float
    buildings: List[Building]
    agent_communicators: Dict[int, List[int]]
    id_houses_messages: List[int]
    communication_builder: AgentCommunicationBuilder

    def __init__(self, cluster_props: ClusterPropreties) -> None:
        """Initialize Cluster."""
        self.init_props = deepcopy(cluster_props)
        self.reset()

    def reset(self) -> List[EnvironmentObsDict]:
        """Reset cluster class.

        Returns: A dictionnary containing the state of the cluster
        """
        self.max_power = 0.0
        self.current_power_consumption = 0.0
        self.id_houses_messages: List[int] = []
        self.buildings = [
            Building(self.init_props.house_prop)
            for _ in range(self.init_props.nb_agents)
        ]
        for building in self.buildings:
            self.current_power_consumption += building.get_power_consumption()
            self.max_power += building.max_consumption
        self.communication_builder = AgentCommunicationBuilder(
            agents_comm_props=self.init_props.agents_comm_prop,
            nb_agents=self.init_props.nb_agents,
        )
        self.agent_communicators = self.communication_builder.get_comm_link_list()
        return self.get_obs()

    def step(
        self,
        od_temp: float,
        action_dict: Dict[int, bool],
        date_time: datetime,
        time_step: timedelta,
    ) -> List[EnvironmentObsDict]:
        """Take a step in time for the cluster given the list of actions of the TCL agent."""
        self.current_power_consumption = 0.0
        for building_id, building in enumerate(self.buildings):
            if building_id in action_dict.keys():
                command = action_dict[building_id]
            else:
                command = False
            building.step(od_temp, time_step, date_time, command)
            self.current_power_consumption += building.get_power_consumption()
        return self.get_obs()

    def message(self, building_id: int) -> List[BuildingMessage]:
        """List of messages sent from the other agents to the building with building_id index.

        If the communication mode chosen in config is random_sample,
        """
        if self.init_props.agents_comm_prop.mode == "random_sample":
            self.id_houses_messages = self.communication_builder.get_random_sample(
                building_id
            )
        else:
            self.id_houses_messages = self.agent_communicators[building_id]

        message: List[BuildingMessage] = []
        for id_house_message in self.id_houses_messages:
            message.append(
                self.buildings[id_house_message].message(
                    self.init_props.message_prop.thermal,
                    self.init_props.message_prop.hvac,
                )
            )
        return message

    def get_obs(self) -> List[EnvironmentObsDict]:
        """Generate cluster observation dictionnary."""
        state_dict: List[EnvironmentObsDict] = []
        for building_id, building in enumerate(self.buildings):
            building_obs = building.get_obs()
            building_obs["cluster_hvac_power"] = self.current_power_consumption
            building_obs["message"] = self.message(building_id)
            state_dict.append(building_obs)
        return state_dict

    def apply_noise(self) -> None:
        """Apply noise to cluster properties."""
        for building in self.buildings:
            building.apply_noise()

import random
from copy import deepcopy
from typing import Dict, List

import numpy as np

from app.core.environment.cluster.cluster_properties import (
    AgentsCommunicationProperties,
)


class AgentCommunicationBuilder:
    """
    Builds a communication graph between agents. The communication graph determines the agents with whom a
    particular agent can communicate during the simulation. There are four communication modes implemented:
        - neighbours: agents can communicate with their closest neighbours in a circular fashion.
        - closed_groups: agents are grouped into sets, each set containing a maximum of max_nb_agents_communication
            agents, and agents can communicate only with agents within the same set.
        - random_fixed: agents are connected to a fixed number of other agents, drawn randomly with replacement
            from the entire set of agents.
        - neighbours_2D: agents can communicate with their neighbours in a 2D grid.


    Attributes:
        agents_comm_props: AgentsCommunicationProperties
            The properties of the agents' communication.
        nb_comm: int
            The maximum number of agents that each agent can communicate with.
        nb_agents: int
            The total number of agents.
        agent_ids: List[int]
            The list of IDs of all agents.

    """

    def __init__(
        self,
        agents_comm_props: AgentsCommunicationProperties,
        nb_agents: int,
    ) -> None:
        """
        Initialize an instance of the AgentCommunicationBuilder class.

        Parameters:
            - agents_comm_props: An instance of the AgentsCommunicationProperties class which stores the properties of the communication between agents.
            - nb_agents: The number of agents in the simulation.
        """
        self.agents_comm_props = agents_comm_props
        self.nb_comm = np.minimum(
            agents_comm_props.max_nb_agents_communication,
            (nb_agents - 1),
        )
        self.nb_agents = nb_agents
        self.agent_ids = list(range(nb_agents))

    def get_comm_link_list(self) -> Dict[int, List[int]]:
        """
        Return a dictionary with the IDs of agents as keys and a list of IDs of agents with which they can communicate as values.
        """
        mode = getattr(self, self.agents_comm_props.mode)
        return mode()

    def neighbours(self) -> Dict[int, List[int]]:
        """
        Get the neighbours of each agent in a circular fashion,
        if agent_id is 5:
            the half before will be [0, 1, 2, 3, 4]
            and half after will be [6, 7, 8, 9, 10]
        if agent_id is 1:
            the half before will be [7, 8, 9, 10, 0]
            and half after will be [2, 3, 4, 5, 6]
        """
        agent_communicators: Dict[int, List[int]] = {}
        for agent_id in self.agent_ids:
            half_before = [
                (agent_id - int(np.floor(self.nb_comm / 2)) + i) % len(self.agent_ids)
                for i in range(int(np.floor(self.nb_comm / 2)))
            ]
            half_after = [
                (agent_id + 1 + i) % len(self.agent_ids)
                for i in range(int(np.ceil(self.nb_comm / 2)))
            ]
            ids_houses_messages = half_before + half_after
            agent_communicators[agent_id] = ids_houses_messages
        return agent_communicators

    def closed_groups(self) -> Dict[int, List[int]]:
        """
        Return a dictionary with the IDs of agents as keys and a list of IDs of agents with which they can communicate within the same group.
        Agents are grouped into sets, each containing a maximum of max_nb_agents_communication agents.
        """
        agent_communicators: Dict[int, List[int]] = {}

        for agent_id in self.agent_ids:
            base = agent_id - (agent_id % (self.nb_comm + 1))
            if base + self.nb_comm <= len(self.agent_ids):
                ids_houses_messages = [
                    base + i
                    for i in range(
                        self.agents_comm_props.max_nb_agents_communication + 1
                    )
                ]
            else:
                ids_houses_messages = [
                    len(self.agent_ids) - self.nb_comm - 1 + i
                    for i in range(self.nb_comm + 1)
                ]
            ids_houses_messages.remove(agent_id)
            agent_communicators[agent_id] = ids_houses_messages
        return agent_communicators

    def random_sample(self) -> Dict[int, List[int]]:
        """
        This method is intended to return a dictionary with the IDs of agents as keys and a list of IDs of
        randomly selected agents with which they can communicate.
        """
        return {}

    def random_fixed(self) -> Dict[int, List[int]]:
        """
        Returns a dictionary with the IDs of agents as keys and a list of IDs of randomly selected agents with which
        they can communicate. Each agent is connected to a fixed number of other agents,
        drawn randomly with replacement from the entire set of agents.
        """
        agent_communicators: Dict[int, List[int]] = {}
        for agent_id in self.agent_ids:
            agent_communicators[agent_id] = self.get_random_sample(agent_id)
        return agent_communicators

    def neighbours_2D(self) -> Dict[int, List[int]]:
        """
        Returns a dictionary with the IDs of agents as keys and a list of IDs of neighbouring agents with which they can communicate in a 2D grid.
        The communication distance is limited by the max_communication_distance parameter and the row size is specified by the row_size parameter.
        """
        if len(self.agent_ids) % self.agents_comm_props.row_size != 0:
            # TODO: put this in validator of model
            raise ValueError("Neighbours 2D row_size must be a divisor of nb_agents")

        max_y = len(self.agent_ids) // self.agents_comm_props.row_size
        if (
            self.agents_comm_props.max_communication_distance
            >= (self.agents_comm_props.row_size + 1) // 2
            or self.agents_comm_props.max_communication_distance >= (max_y + 1) // 2
        ):
            # TODO: put this in validator of model
            raise ValueError(
                "Neighbours 2D distance_comm ({}) must be strictly smaller than (row_size+1) / 2 ({}) and (max_y+1) / 2 ({})".format(
                    self.agents_comm_props.max_communication_distance,
                    (self.agents_comm_props.row_size + 1) // 2,
                    (max_y + 1) // 2,
                )
            )

        distance_pattern = []
        for x_diff in range(
            -1 * self.agents_comm_props.max_communication_distance,
            self.agents_comm_props.max_communication_distance + 1,
        ):
            for y_diff in range(
                -1 * self.agents_comm_props.max_communication_distance,
                self.agents_comm_props.max_communication_distance + 1,
            ):
                if abs(x_diff) + abs(
                    y_diff
                ) <= self.agents_comm_props.max_communication_distance and (
                    x_diff != 0 or y_diff != 0
                ):
                    distance_pattern.append((x_diff, y_diff))
        agent_communicators: Dict[int, List[int]] = {}

        for agent_id in self.agent_ids:
            x = agent_id % self.agents_comm_props.row_size
            y = agent_id // self.agents_comm_props.row_size
            ids_houses_messages = []
            for pair_diff in distance_pattern:
                x_new = x + pair_diff[0]
                y_new = y + pair_diff[1]
                if x_new < 0:
                    x_new += self.agents_comm_props.row_size
                if x_new >= self.agents_comm_props.row_size:
                    x_new -= self.agents_comm_props.row_size
                if y_new < 0:
                    y_new += max_y
                if y_new >= max_y:
                    y_new -= max_y
                agent_id_new = y_new * self.agents_comm_props.row_size + x_new
                ids_houses_messages.append(agent_id_new)
            agent_communicators[agent_id] = ids_houses_messages
        return agent_communicators

    def get_random_sample(self, agent_id) -> List[int]:
        """
        Returns a list of `nb_comm` randomly selected agent IDs, excluding the given `agent_id`.

        Parameters:
            agent_id (int): The ID of the agent to exclude from the possible IDs that can be selected.

        Returns:
            List[int]: A list of `nb_comm` randomly selected agent IDs, excluding the given `agent_id`.
        """
        possible_ids = deepcopy(self.agent_ids)
        possible_ids.remove(agent_id)
        return random.sample(possible_ids, k=self.nb_comm)

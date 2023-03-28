from copy import deepcopy
import random
from typing import Dict, List

import numpy as np
from app.core.environment.cluster.cluster_properties import (
    AgentsCommunicationProperties,
)


class AgentCommunicationBuilder:
    def __init__(
        self,
        agents_comm_props: AgentsCommunicationProperties,
        nb_agents: int,
    ) -> None:
        self.agents_comm_props = agents_comm_props
        self.nb_comm = np.minimum(
            agents_comm_props.max_nb_agents_communication,
            (nb_agents - 1),
        )
        self.nb_agents = nb_agents
        self.agent_ids = list(range(nb_agents))

    def get_comm_link_list(self) -> Dict[int, List[int]]:
        mode = {
            "neighbours": self.neighbours(),
            "closed_groups": self.closed_groups(),
            "random_sample": self.random_sample(),
            "random_fixed": self.random_fixed(),
            "neighbours_2D": self.neighbours_2D(),
        }
        return mode[self.agents_comm_props.mode]

    def neighbours(self) -> Dict[int, List[int]]:
        """
        Gets the neighbours of each agent in a circular fashion,
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
        return {}

    def random_fixed(self) -> Dict[int, List[int]]:
        agent_communicators: Dict[int, List[int]] = {}
        for agent_id in self.agent_ids:
            agent_communicators[agent_id] = self.get_random_sample(agent_id)
        return agent_communicators

    def neighbours_2D(self) -> Dict[int, List[int]]:
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
        possible_ids = deepcopy(self.agent_ids)
        possible_ids.remove(agent_id)
        return random.sample(possible_ids, k=self.nb_comm)

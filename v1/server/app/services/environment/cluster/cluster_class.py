from typing import List

from v1.server.app.services.environment.cluster.buildings.building import Building
from v1.server.app.services.environment.cluster.cluster_properties import (
    ClusterProperties,
    TemperatureProperties,
)


class BuildingCluster:

    buildings: List[Building]
    cluster_properties: ClusterProperties()
    temperature_properties: TemperatureProperties()

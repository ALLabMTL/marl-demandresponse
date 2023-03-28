from typing import Dict, List
from app.core.environment.cluster.building import Building
from app.core.environment.environment_properties import RewardProperties
from app.utils.utils import deadbandL2
from app.core.environment.cluster.building_properties import BuildingProperties


class RewardsCalculator:
    def __init__(
        self, reward_props: RewardProperties, building_props: BuildingProperties
    ) -> None:
        self.reward_props = reward_props
        self.building_props = building_props

    def compute_temp_penalty(
        self, building_id: int, buildings: List[Building]
    ) -> float:
        """
        Returns: a float, representing the positive penalty due to distance between the target (indoors) temperature and the indoors temperature in a house.

        Parameters:
        building_id: an int, representing the index of the building we want to compute the penalty.
        buildings: the list of Buildings in the environment, useful to compute using common l2, common max and mixture methods.
        """
        modes = {
            "individual_L2": self.individual_l2(buildings[building_id]),
            "common_L2": self.common_l2(buildings),
            "common_max": self.common_max(buildings),
            "mixture": self.mixture(building_id, buildings),
        }
        return modes[self.reward_props.penalty_props.mode]

    def common_l2(self, buildings: List[Building]) -> float:
        temperature_penalty = 0.0
        for building in buildings:
            building_temperature_penalty = deadbandL2(
                building.initial_properties.target_temp,
                building.initial_properties.deadband,
                building.indoor_temp,
            )
            temperature_penalty += building_temperature_penalty / len(buildings)
        return temperature_penalty

    def individual_l2(self, building: Building) -> float:
        temperature_penalty = deadbandL2(
            building.initial_properties.target_temp,
            building.initial_properties.deadband,
            building.indoor_temp,
        )
        return temperature_penalty

    def common_max(self, buildings: List[Building]) -> float:
        temperature_penalty = 0.0
        for building in buildings:
            house_temperature_penalty = deadbandL2(
                building.initial_properties.target_temp,
                building.initial_properties.deadband,
                building.indoor_temp,
            )
            if house_temperature_penalty > temperature_penalty:
                temperature_penalty = house_temperature_penalty
        return temperature_penalty

    def mixture(self, building_id: int, buildings: List[Building]) -> float:
        common_l2 = self.common_l2(buildings)
        common_max = self.common_max(buildings)
        ind_l2 = self.individual_l2(buildings[building_id])

        ## Putting together
        alpha_ind_l2 = self.reward_props.penalty_props.alpha_ind_l2
        alpha_common_l2 = self.reward_props.penalty_props.alpha_common_l2
        alpha_common_max = self.reward_props.penalty_props.alpha_common_max
        temperature_penalty = (
            alpha_ind_l2 * ind_l2
            + alpha_common_l2 * common_l2
            + alpha_common_max * common_max
        ) / (alpha_ind_l2 + alpha_common_l2 + alpha_common_max)
        return temperature_penalty

    def compute_rewards(
        self,
        buildings: List[Building],
        cluster_hvac_power: float,
        power_grid_reg_signal: float,
    ) -> Dict[int, float]:
        """
        Compute the reward of each TCL agent

        Returns:
        rewards_dict: a dictionary, containing the rewards of each TCL agent.

        Parameters:
        temp_penalty_dict: a dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """
        rewards_dict: Dict[int, float] = {}
        signal_penalty = self.reg_signal_penalty(
            cluster_hvac_power, power_grid_reg_signal, len(buildings)
        )

        norm_temp_penalty = deadbandL2(
            self.building_props.target_temp,
            0,
            self.building_props.target_temp + 1,
        )

        norm_sig_penalty = deadbandL2(
            self.reward_props.norm_reg_signal,
            0,
            0.75 * self.reward_props.norm_reg_signal,
        )

        # Temperature penalties
        temp_penalty_dict: Dict[int, float] = {}
        for building_id, _ in enumerate(buildings):
            temp_penalty_dict[building_id] = self.compute_temp_penalty(
                building_id, buildings
            )
            rewards_dict[building_id] = -1 * (
                self.reward_props.alpha_temp
                * temp_penalty_dict[building_id]
                / norm_temp_penalty
                + self.reward_props.alpha_sig * signal_penalty / norm_sig_penalty
            )
        return rewards_dict

    def reg_signal_penalty(
        self, cluster_hvac_power: float, power_grid_reg_signal: float, nb_agents: int
    ) -> float:
        """
        Returns: a float, representing the positive penalty due to the distance between the regulation signal and the total power used by the TCLs.

        Parameters:
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        nb_agents: an int, representing the number of agents in the cluster
        """
        # TODO: change config to have a signal penalty mode too
        # add validators in config parser
        if self.reward_props.penalty_props.mode == "common_L2":
            penalty = ((cluster_hvac_power - power_grid_reg_signal) / nb_agents) ** 2
        else:
            raise ValueError(
                f"Unknown signal penalty mode: {self.reward_props.penalty_props.mode}"
            )

        return penalty

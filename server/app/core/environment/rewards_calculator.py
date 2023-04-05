from typing import Dict, List

from app.core.environment.cluster.building import Building
from app.core.environment.environment_properties import (
    BuildingProperties,
    RewardProperties,
)
from app.utils.utils import deadbandL2


class RewardsCalculator:
    """
    Calculates rewards for TCL agents based on temperature penalties and regulation signal penalties.

    Atributes:
        reward_props (RewardProperties): Properties used for reward computation.
        building_props (BuildingProperties): Properties used for building simulation.
    """

    def __init__(
        self, reward_props: RewardProperties, building_props: BuildingProperties
    ) -> None:
        """
        It sets the instance variables self.reward_props and self.building_props to these arguments. The __init__ method is called when a new instance of this class is created.
        """
        self.reward_props = reward_props
        self.building_props = building_props

    def compute_temp_penalty(
        self, building_id: int, buildings: List[Building]
    ) -> float:
        """
        Computes the temperature penalty.

        Parameters:
            - building_id: an int, representing the index of the building we want to compute the penalty.
            - buildings: the list of Buildings in the environment, useful to compute using common l2, common max and mixture methods.
        
        Returns: 
            float: representing the positive penalty due to distance between the target (indoors) temperature and the indoors temperature in a house.

        """
        mode = getattr(self, self.reward_props.penalty_props.mode)
        return mode(building_id, buildings)

    def common_L2(self, building_id: int, buildings: List[Building]) -> float:
        """
        Computes the average deadband penalty across all buildings in the simulation.

        Parameters:
            - building_id (int): ID of the building being evaluated.
            - buildings (List[Building]): List of all buildings in the simulation.

        Returns:
            float: The average deadband penalty across all buildings.
        """
        temperature_penalty = 0.0
        for building in buildings:
            building_temperature_penalty = deadbandL2(
                building.init_props.target_temp,
                building.init_props.deadband,
                building.indoor_temp,
            )
            temperature_penalty += building_temperature_penalty / len(buildings)
        return temperature_penalty

    def individual_L2(self, building_id: int, buildings: List[Building]) -> float:
        """
        Computes the L2 penalty for a specific building, based on the temperature deviation from the target temperature.

        Parameters:
            - building_id (int): Index of the building to compute the L2 penalty for.
            - buildings (List[Building]): List of Building objects representing the buildings.

        Returns:
            float: The L2 penalty for the specified building.
        """
        building = buildings[building_id]
        temperature_penalty = deadbandL2(
            building.init_props.target_temp,
            building.init_props.deadband,
            building.indoor_temp,
        )
        return temperature_penalty

    def common_max_error(self, building_id: int, buildings: List[Building]) -> float:
        """
        Computes the maximum temperature deviation penalty across all buildings in the simulation.

        Parameters:
            - building_id (int): Index of the building to exclude from the computation.
            - buildings (List[Building]): List of Building objects representing the buildings.

        Returns:
            float: The maximum temperature deviation penalty across all buildings in the simulation.
        """
        temperature_penalty = 0.0
        for building in buildings:
            house_temperature_penalty = deadbandL2(
                building.init_props.target_temp,
                building.init_props.deadband,
                building.indoor_temp,
            )
            if house_temperature_penalty > temperature_penalty:
                temperature_penalty = house_temperature_penalty
        return temperature_penalty

    def mixture(self, building_id: int, buildings: List[Building]) -> float:
        """
        Computes a penalty mixture for a specific building based on individual and common L2 penalties and the maximum
        temperature deviation penalty across all buildings.

        Parameters:
            - building_id (int): Index of the building to compute the penalty mixture for.
            - buildings (List[Building]): List of Building objects representing the buildings.

        Returns:
            float: The temperature penalty mixture for the specified building.
        """
        common_l2 = self.common_L2(building_id, buildings)
        common_max = self.common_max_error(building_id, buildings)
        ind_l2 = self.individual_L2(building_id, buildings)

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
            - temp_penalty_dict: a dictionary, containing the temperature penalty for each TCL agent
            - cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
            - power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
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
            self.reward_props.norm_reg_sig,
            0,
            0.75 * self.reward_props.norm_reg_sig,
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
            - cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
            - power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
            - nb_agents: an int, representing the number of agents in the cluster
        """
        # TODO: change config to have a signal penalty mode too
        # add validators in config parser
        if self.reward_props.sig_penalty_mode == "common_L2":
            penalty = ((cluster_hvac_power - power_grid_reg_signal) / nb_agents) ** 2
        else:
            raise ValueError(
                f"Unknown signal penalty mode: {self.reward_props.penalty_props.mode}"
            )

        return penalty

import csv
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import numpy as np
from scipy.interpolate import interpn

from app.core.environment.cluster.building import Building
from app.core.environment.environment_properties import BuildingProperties
from app.core.environment.power_grid.power_grid_properties import BasePowerProperties
from app.utils.utils import sort_dict_keys

sys.path.insert(1, "../marl-demandresponse")


SECOND_IN_A_HOUR = 3600
NB_TIME_STEPS_BY_SIM = 450


class PowerInterpolator(object):
    """
    Class that allows to interpolate the power demand based on the provided data.

    Attributes:
    -----------
    base_power_props : BasePowerProperties
        An instance of the `BasePowerProperties` class, which contains information about the power grid properties.
    parameters_dict : Dict[str, List[float]]
        A dictionary that contains the values of each parameter used in the power demand model.
    dict_keys : List[str]
        A list that contains the keys of the `parameters_dict` dictionary.
    nb_params : List[int]
        A list that contains the number of values for each parameter in the `parameters_dict` dictionary.
    nb_params_multi : List[int]
        A list that contains the multiplication factor for each parameter in the `parameters_dict` dictionary.
    points : List[np.ndarray]
        A list of NumPy arrays containing the coordinates of each point in the parameter space.
    dimensions_array : List[int]
        A list that contains the number of dimensions for each parameter in the `parameters_dict` dictionary.
    values : np.ndarray
        A NumPy array that contains the power demand values for each point in the parameter space.
    default_building_props : BuildingProperties
        An instance of the `BuildingProperties` class, which contains information about the building properties.
    """

    def __init__(
        self,
        base_power_props: BasePowerProperties,
        default_building_props: BuildingProperties,
    ) -> None:
        """
        Constructor for the PowerInterpolator class.

        Parameters
            base_power_props (BasePowerProperties): Object containing information about the power grid and data files.
            default_building_props (BuildingProperties): Object containing information about the building.

        Returns
            None
        """
        self.base_power_props = base_power_props
        with open(base_power_props.path_parameter_dict) as json_file:
            self.parameters_dict = json.load(json_file)
        with open(base_power_props.path_dict_keys) as f:
            reader = csv.reader(f)
            self.dict_keys = list(reader)[0]

        self.nb_params = []
        for key in self.dict_keys:
            self.nb_params.append(len(self.parameters_dict[key]))

        self.nb_params_multi: List[int] = []
        combined_nb = 1
        for nb_param in reversed(self.nb_params):
            self.nb_params_multi = [combined_nb] + self.nb_params_multi
            combined_nb *= nb_param

        self.points = list(self.parameters_dict.values())

        self.dimensions_array = [
            len(self.parameters_dict[key]) for key in self.dict_keys
        ]

        self.values = np.load(base_power_props.path_datafile).reshape(
            *self.dimensions_array, 1
        )
        self.default_building_props = default_building_props

    def param2index(self, point_dict):
        """
        Converts a dictionary of power parameters to an index that can be used to extract power data from the loaded data files.

        Parameters
            point_dict (Dict[str, Any]): A dictionary containing the parameters to be converted to an index.

        Returns
            index_df (int): An integer representing the index for the provided power parameters.
        """
        "Return the index for a given set of parameters. ! If date in point_dict, must be the day # in the year (timetuple().tm_yday property of datetime)"
        assert point_dict.keys() == self.dict_keys

        values = list(point_dict.values())

        base_values = list(self.parameters_dict.values())  # list of lists
        indices = []
        for i in range(len(values)):
            value = values[i]
            list_values = base_values[i]
            idx = list_values.index(value)
            indices.append(idx)

        index_df = 0
        for i in range(len(indices)):
            index_df += indices[i] * self.nb_params_multi[i]

        return index_df

    def interpolate_grid(self, point_dict):
        """
        Interpolates the power demand for a given set of power parameters using the loaded data files.

        Parameters
            point_dict (Dict[str, Any]): A dictionary containing the power parameters to be used for interpolation.

        Returns
            result (float): The interpolated power demand value.
        """
        point_coordinates = list(point_dict.values())
        result = interpn(self.points, self.values, point_coordinates)
        # print(result)
        return result

    def interpolate_grid_fast(self, point_dict) -> float:
        """
        Returns a fast interpolation, using nearest neighbour for the house thermal parameters and linear interpolation for the other parameters.

        Parameters
            point_dict (Dict[str, Any]): A dictionary containing the power parameters to be used for interpolation.

        Returns
            result (float): The interpolated power demand value.
        """
        point_coordinates = list(point_dict.values())[
            4:
        ]  # Remove the house thermal parameters
        points = self.points[4:]  # Remove the house thermal parameters

        # Thermal parameters
        closest_id = []
        for i in range(4):
            value = list(point_dict.values())[i]
            distances = np.abs(np.array(self.points[i]) - value)
            index = np.argmin(distances)
            closest_id.append(index)

        values_therm = self.values[closest_id[0]][closest_id[1]][closest_id[2]][
            closest_id[3]
        ]

        # HVAC power
        hvac_power = point_dict["HVAC_power"]
        hvac_power_points = self.parameters_dict["HVAC_power"]
        index_hvac = np.argmin(np.abs(np.array(hvac_power_points) - hvac_power))

        values_cut = values_therm[
            :, :, :, index_hvac, :, :
        ]  # air_temp, mass_temp, OD_temp, HVAC_power, hour, date
        points_cut = deepcopy(points)
        del points_cut[3]
        del point_coordinates[3]

        result = interpn(points_cut, values_cut, point_coordinates)[0][0]
        # print(result)
        return result

    def get_two_closest(self, array, value):
        "Given a value, return the closest values from a sorted numpy array. If value is inside the list range, return one lower and one higher. If all are lower or higher, return the two extreme values."
        distances = np.abs(array - value)
        indices_two_closest = np.argsort(distances)[:2]
        return array[indices_two_closest]

    def interpolate_power(
        self,
        date_time: datetime,
        current_od_temp: float,
        interp_nb_agents,
        buildings: List[Building],
    ) -> float:
        """
        Given a value, returns the closest values from a sorted numpy array. If value is inside the list range, return one lower and one higher. If all are lower or higher, return the two extreme values.

        Parameters
            array (numpy.ndarray): A sorted numpy array.
            value (float): The value for which to find the closest values.
        Returns
            closest (numpy.ndarray): An array containing the two closest values.
        """
        base_power = 0.0

        if self.default_building_props.solar_gain:
            point = {
                "date": date_time.timetuple().tm_yday,
                "hour": (
                    date_time
                    - date_time.replace(hour=0, minute=0, second=0, microsecond=0)
                ).total_seconds(),
            }
        else:  # No solar gain - make it think it is midnight
            point = {
                "date": 0.0,
                "hour": 0.0,
            }

        all_ids = list(range(len(buildings)))
        if len(all_ids) <= interp_nb_agents:
            interp_house_ids = all_ids
            multi_factor = 1.0
        else:
            interp_house_ids = random.choices(all_ids, k=interp_nb_agents)
            multi_factor = float(len(all_ids)) / float(interp_nb_agents)
        # Adding the interpolated power for each house
        for house_id in interp_house_ids:
            house = buildings[house_id]
            point["Ua_ratio"] = (
                house.init_props.Ua / self.default_building_props.Ua
            )  # TODO: This is ugly as in the Monte Carlo, we compute the ratio based on the Ua in config. We should change the dict for absolute numbers.
            point["Cm_ratio"] = house.init_props.Cm / self.default_building_props.Cm
            point["Ca_ratio"] = house.init_props.Ca / self.default_building_props.Ca
            point["Hm_ratio"] = house.init_props.Hm / self.default_building_props.Hm
            point["air_temp"] = house.indoor_temp - house.init_props.target_temp
            point["mass_temp"] = house.current_mass_temp - house.init_props.target_temp
            point["OD_temp"] = current_od_temp - house.init_props.target_temp
            point["HVAC_power"] = house.hvac.init_props.cooling_capacity
            point = self.clip_interpolation_point(point)
            point = sort_dict_keys(point, self.dict_keys)
            base_power += self.interpolate_grid_fast(point)
        base_power *= multi_factor

        return base_power

    def clip_interpolation_point(self, point: Dict[str, float]) -> Dict[str, float]:
        """
        Interpolates the power demand for a given set of power parameters using the loaded data files.

        Parameters
            date_time (datetime): A datetime object containing the current date and time.
            current_od_temp (float): The current outdoor temperature.
            interp_nb_agents (int): The number of buildings to use for interpolation.
            buildings (List[Building]): A list of Building objects representing the buildings in the simulation.

        Returns
            result (float): The interpolated power demand value.
        """
        for key in point.keys():
            values = np.array(self.parameters_dict[key])
            if point[key] > np.max(values):
                point[key] = np.max(values)
            elif point[key] < np.min(values):
                point[key] = np.min(values)
        return point


# if __name__ == "__main__":
# parameters_dict = {
#     "Ua_ratio": [1, 1.1],
#     "Cm_ratio": [1, 1.1],
#     "Ca_ratio": [1, 1.1],
#     "Hm_ratio": [1, 1.1],
#     "air_temp": [0, 1],
#     "mass_temp": [0, 1],
#     "OD_temp": [11, 13],
#     "HVAC_power": [10000, 15000],
#     "hour": [0.0, 10800.0],
#     "date": [0, 79],
# }

# dict_keys = [
#     "Ua_ratio",
#     "Cm_ratio",
#     "Ca_ratio",
#     "Hm_ratio",
#     "air_temp",
#     "mass_temp",
#     "OD_temp",
#     "HVAC_power",
#     "hour",
#     "date",
# ]

# power_inter = PowerInterpolator(
#     "./mergedGridSearchResultFinal.npy", parameters_dict, dict_keys
# )

# try_0 = {
#     "Ua_ratio": 1.1,
#     "Cm_ratio": 1.1,
#     "Ca_ratio": 1.1,
#     "Hm_ratio": 1.1,
#     "air_temp": 0,
#     "mass_temp": 0,
#     "OD_temp": 13,
#     "HVAC_power": 10000,
#     "hour": 0,
#     "date": 0,
# }

# id0 = power_inter.interpolate_grid_fast(try_0)
# id1 = power_inter.interpolate_grid(try_0)

# print("Fast: {}".format(id0))
# print("Normal: {}".format(id1))

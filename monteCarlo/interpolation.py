import sys

sys.path.insert(1, "../marl-demandresponse")

import copy
import datetime
import itertools as it
import time
from copy import deepcopy

import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interpn
import numpy as np

SECOND_IN_A_HOUR = 3600
NB_TIME_STEPS_BY_SIM = 450

class PowerInterpolator(object):

	def __init__(self, path, parameters_dict, dict_keys):

		self.parameters_dict = parameters_dict  # All parameters used in the dataframe
		self.dict_keys = dict_keys

		self.nb_params = []
		for key in self.dict_keys:
			self.nb_params.append(len(self.parameters_dict[key]))


		self.nb_params_multi = []
		combined_nb = 1
		for nb_param in reversed(self.nb_params):
			self.nb_params_multi = [combined_nb] + self.nb_params_multi
			combined_nb *= nb_param

		self.points = list(self.parameters_dict.values())

		self.dimensions_array = [len(self.parameters_dict[key]) for key in self.dict_keys]

		self.values = np.load(path).reshape(*self.dimensions_array,1)


	def param2index(self, point_dict):
		"Return the index for a given set of parameters. ! If date in point_dict, must be the day # in the year (timetuple().tm_yday property of datetime)"
		assert point_dict.keys() == self.dict_keys


		values = list(point_dict.values())	

		base_values = list(self.parameters_dict.values()) # list of lists
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

	def interpolateLinearND(self, point_dict):
		"Return a linear interpolation for a point"
		assert set(point_dict.keys()) == set(self.dict_keys)

		point_coordinates = list(point_dict.values())
		
		closest_coordinates_grid = {}
		# Getting highest and lowest points for each coordinates
		for key in self.dict_keys:
			closest_coordinates_grid[key] = self.get_two_closest(np.array(parameters_dict[key]), point_dict[key])


		# Getting the coordinates of all points on the grid that we want to give to the interpolator
		list_coordinates = list(closest_coordinates_grid.values())
		all_combinations = np.array(np.meshgrid(*list_coordinates)).T.reshape(-1,len(point_coordinates)) # Gives a list of point coordinates representing all combinations from the two points per dimension.
		all_combinations_list = all_combinations.tolist()

		## Getting the values
		power_list = []
		for coordinates in all_combinations_list:
			point = {}
			i = 0
			for key in self.dict_keys:
				point[key] = coordinates[i]
				i += 1
			index_df = self.param2index(point)
			power = self.power_data.loc[[index_df]]["hvac_average_power"]
			power_list.append(power)
		power_array = np.array(power_list)

		start_time = time.time()

		print("Ready to start interpolator")

		interp = LinearNDInterpolator(all_combinations, power_array)
		print("Interpolator loaded. Time for loading: ", str(datetime.timedelta(seconds=round(time.time() - start_time))))
		print("Interpolating...")

		result = interp(point_coordinates)
		
		print("Done! Time for loading + interpolating: ", str(datetime.timedelta(seconds=round(time.time() - start_time))),)
		return result

	def interpolateGrid(self, point_dict):
		point_coordinates = list(point_dict.values())
		result = interpn(self.points, self.values, point_coordinates)
		#print(result)
		return result

	def interpolateGridFast(self, point_dict):
		""" 
		Returns a fast interpolation, using nearest neighbour for the house thermal parameters and linear interpolation for the other parameters. 
		"""
		point_coordinates = list(point_dict.values())[4:]		# Remove the house thermal parameters
		points = self.points[4:]								# Remove the house thermal parameters

		# Thermal parameters
		closest_id = []
		for i in range(4):
			value = list(point_dict.values())[i]
			distances = np.abs(np.array(self.points[i]) - value)
			index = np.argmin(distances)
			closest_id.append(index)

		values_therm = self.values[closest_id[0]][closest_id[1]][closest_id[2]][closest_id[3]]

		# HVAC power
		hvac_power = point_dict["HVAC_power"]
		hvac_power_points = self.parameters_dict["HVAC_power"]
		index_hvac = np.argmin(np.abs(np.array(hvac_power_points) - hvac_power))

		values_cut = values_therm[:,:,:,index_hvac,:,:]				# air_temp, mass_temp, OD_temp, HVAC_power, hour, date
		points_cut = deepcopy(points)
		del points_cut[3]
		del point_coordinates[3]

		result = interpn(points_cut, values_cut, point_coordinates)
		#print(result)
		return result


	def get_two_closest(self, array, value):
		"Given a value, return the closest values from a sorted numpy array. If value is inside the list range, return one lower and one higher. If all are lower or higher, return the two extreme values."
		distances = np.abs(array - value)
		indices_two_closest = np.argsort(distances)[:2]
		return array[indices_two_closest]

if __name__ == "__main__":

	parameters_dict = {"Ua_ratio": [1, 1.1], "Cm_ratio": [1, 1.1], "Ca_ratio": [1, 1.1], "Hm_ratio": [1, 1.1], "air_temp": [0, 1], "mass_temp": [0, 1], "OD_temp": [11, 13], "HVAC_power": [10000, 15000], "hour": [0.0, 10800.0], "date": [0, 79]}


	dict_keys = ["Ua_ratio", "Cm_ratio", "Ca_ratio", "Hm_ratio", "air_temp", "mass_temp", "OD_temp", "HVAC_power", "hour", "date"]

	power_inter = PowerInterpolator('./mergedGridSearchResultFinal.npy', parameters_dict, dict_keys)


	try_0 = {
	"Ua_ratio": 1.1,
	"Cm_ratio": 1.1,
	"Ca_ratio": 1.1,
	"Hm_ratio": 1.1,
	"air_temp": 0,
	"mass_temp": 0,
	"OD_temp": 13,
	"HVAC_power": 10000,
	"hour": 0,
	"date": 0,
	}


	id0 = power_inter.interpolateGridFast(try_0)
	id1 = power_inter.interpolateGrid(try_0)

	print("Fast: {}".format(id0))
	print("Normal: {}".format(id1))



"""
	try_0 = {
	"Ua_ratio": 1,
	"Cm_ratio": 1,
	"Ca_ratio": 0.9,
	"Hm_ratio": 0.9,
	"air_temp": -4,
	"mass_temp": -4,
	"OD_temp": 3,
	"HVAC_power": 10000,
	"hour": 3,
	"date": 83,
	}
	try_1 = {
	"Ua_ratio": 1,
	"Cm_ratio": 1,
	"Ca_ratio": 0.9,
	"Hm_ratio": 0.9,
	"air_temp": 3,
	"mass_temp": -4,
	"OD_temp": 3,
	"HVAC_power": 10000,
	"hour": 3,
	"date": 83,
	}
	try_2 = {
	"Ua_ratio": 1,
	"Cm_ratio": 1,
	"Ca_ratio": 0.9,
	"Hm_ratio": 0.9,
	"air_temp": 3,
	"mass_temp": 3,
	"OD_temp": 8,
	"HVAC_power": 15000,
	"hour": 13,
	"date": 186,
	}
	try_3 = {
	"Ua_ratio": 1,
	"Cm_ratio": 1,
	"Ca_ratio": 0.9,
	"Hm_ratio": 0.9,
	"air_temp": 3,
	"mass_temp": 3,
	"OD_temp": 8,
	"HVAC_power": 15000,
	"hour": 23,
	"date": 186,
	}
	t1_start = time.process_time() 
	for i in range(500):
		id0 = power_inter.interpolateGridFast(try_0)
		id0 = power_inter.interpolateGridFast(try_1)
		id0 = power_inter.interpolateGridFast(try_2)
		id0 = power_inter.interpolateGridFast(try_2)

	#run your code
	t1_stop = time.process_time()
	
	print("Elapsed time for 2000 interpolations:", t1_stop - t1_start) 
	#id0 = power_inter.param2index(try_0)
	#print("id0: {}".format(id0))

	#points = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 2, 0, 0], [0, 2, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0],[3, 0, 0, 0, -1, 2, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
	#values = np.array([0, 0.5, 1, 3, 5, 1, 2, 0, 2, 1, 3, 5])
	#interp = LinearNDInterpolator(points, values)
	#a = interp([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	#print(a)



	#array = np.array([1, 2, 3, 4, 5, 6])
	#value = -0.1
	#print(get_two_closest(array,value))
"""
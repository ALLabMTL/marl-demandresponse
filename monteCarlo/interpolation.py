import sys

sys.path.insert(1, "../marl-demandresponse")

import copy
import datetime
import itertools as it
import time

import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interpn
import numpy as np



class PowerInterpolator(object):

	def __init__(self, path, parameters_dict, dict_keys):

		print("Loading file...")
		self.power_data = pd.read_csv(path)
		print("File loaded.")
		print("Transforming...")


		## Changing dates to numbers
		self.power_data['date'] = self.power_data.date.apply(lambda x: datetime.datetime.strptime(x, '(%Y, %m, %d)').timetuple().tm_yday)
		print("Transformation done")

		## Converting dates and hours in sin and cos (TBD later)
		"""
		self.power_data['hour_sin'] = self.power_data.apply(lambda row: np.sin(row.hour/24*2*np.pi), axis = 1)
		print("Transforming 2/4...")
		self.power_data['hour_cos'] = self.power_data.apply(lambda row: np.cos(row.hour/24*2*np.pi), axis = 1)
		print("Transforming 3/4...")
		self.power_data['date_sin'] = self.power_data.apply(lambda row: np.sin(row.date/365*2*np.pi), axis = 1)
		print("Transforming 4/4...")
		self.power_data['date_cos'] = self.power_data.apply(lambda row: np.cos(row.date/365*2*np.pi), axis = 1)
		print("Transformed")
		"""

		self.parameters_dict = parameters_dict  # All parameters used in the dataframe
		self.dict_keys = dict_keys
		dates_nb = []
		for date in self.parameters_dict["date"]:
			date = str(date)
			dates_nb.append(datetime.datetime.strptime(date, '(%Y, %m, %d)').timetuple().tm_yday)
		self.parameters_dict["date"] = dates_nb


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

		self.values = self.power_data["hvac_average_power"].to_numpy().reshape(*self.dimensions_array,1)



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

		print(all_combinations.shape)

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

		closest_id = []
		for i in range(4):
			value = list(point_dict.values())[i]
			distances = np.abs(np.array(self.points[i]) - value)
			index = np.argmin(distances)
			closest_id.append(index)

		result = interpn(points, self.values[closest_id[0]][closest_id[1]][closest_id[2]][closest_id[3]], point_coordinates)
		#print(result)
		return result


	def get_two_closest(self, array, value):
		"Given a value, return the closest values from a sorted numpy array. If value is inside the list range, return one lower and one higher. If all are lower or higher, return the two extreme values."
		distances = np.abs(array - value)
		indices_two_closest = np.argsort(distances)[:2]
		return array[indices_two_closest]

if __name__ == "__main__":
	parameters_dict = {
    "Ua": [0.9, 1, 1.1],
    "Cm": [0.9, 1, 1.1],
    "Ca": [0.9, 1, 1.1],
    "Hm": [0.9, 1, 1.1],
    "air_temp": [-4, -2, -1, 0, 1, 2, 4],
    "mass_temp": [-4, -2, 0, 2, 4],
    "OD_temp": [3, 5, 7, 9, 11],
    "HVAC_power": [10000, 15000, 20000],
    "hour": [3, 6, 8, 11, 13, 16, 18, 21],
    "date": [(2021, 3, 21), (2021, 6, 21), (2021, 9, 21), (2021, 12, 21)],
	}

	dict_keys = ["Ua", "Cm", "Ca", "Hm", "air_temp", "mass_temp", "OD_temp", "HVAC_power", "hour", "date"]
	power_inter = PowerInterpolator('./grid_search_result.csv', parameters_dict, dict_keys)

	try_0 = {
    "Ua": 1,
    "Cm": 1,
    "Ca": 0.9,
    "Hm": 0.9,
    "air_temp": -4,
    "mass_temp": -4,
    "OD_temp": 3,
    "HVAC_power": 10000,
    "hour": 3,
    "date": 83,
	}
	try_1 = {
    "Ua": 1,
    "Cm": 1,
    "Ca": 0.9,
    "Hm": 0.9,
    "air_temp": 3,
    "mass_temp": -4,
    "OD_temp": 3,
    "HVAC_power": 10000,
    "hour": 3,
    "date": 83,
	}
	try_2 = {
    "Ua": 1,
    "Cm": 1,
    "Ca": 0.9,
    "Hm": 0.9,
    "air_temp": 3,
    "mass_temp": 3,
    "OD_temp": 8,
    "HVAC_power": 15000,
    "hour": 13,
    "date": 186,
	}
	try_3 = {
    "Ua": 1,
    "Cm": 1,
    "Ca": 0.9,
    "Hm": 0.9,
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

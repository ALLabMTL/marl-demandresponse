from interpolation import PowerInterpolator
import unittest
import pandas as pd
import datetime
from copy import deepcopy
import numpy as np

class TestPowerInterpolator(unittest.TestCase):

	def setUp(self):

		path = './grid_search_result.csv'
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


		self.power_data = pd.read_csv('./grid_search_result.csv')
		self.parameters_dict = deepcopy(parameters_dict)

		self.dict_keys = ["Ua", "Cm", "Ca", "Hm", "air_temp", "mass_temp", "OD_temp", "HVAC_power", "hour", "date"]
		self.power_interp = PowerInterpolator(path, parameters_dict, self.dict_keys)

		dates_nb = []


		for date in self.parameters_dict["date"]:
			date = str(date)
			dates_nb.append(datetime.datetime.strptime(date, '(%Y, %m, %d)').timetuple().tm_yday)
			self.parameters_dict["date"] = dates_nb


		self.num_comb = 1 
		for key in self.dict_keys:
			self.num_comb *= len(self.parameters_dict[key])
	
	# Tests that interpolation on exact data points gives the exact value
	def testExactDataPoints(self):
		for index_df in range(1, self.num_comb, int(self.num_comb/50)):
			gt_power = self.power_data["hvac_average_power"].loc[[index_df]].values
			point = {}
			for key in self.dict_keys:
				point[key] = self.power_data[key].loc[[index_df]].values.item()

			point["date"] = datetime.datetime.strptime(str(point["date"]), '(%Y, %m, %d)').timetuple().tm_yday

			interp_value = self.power_interp.interpolateGridFast(point)

			print("Id: {} - point: {} -- gt: {}, interp: {}".format(index_df, point, gt_power, interp_value))
			self.assertEqual(gt_power, interp_value)



if __name__ == '__main__':
	unittest.main()
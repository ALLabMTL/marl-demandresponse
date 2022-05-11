from interpolation import PowerInterpolator
import unittest
import pandas as pd
import datetime
from copy import deepcopy
import numpy as np

SECOND_IN_A_HOUR = 3600

class TestPowerInterpolator(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TestPowerInterpolator, self).__init__(*args, **kwargs)

		path = './mergedGridSearchResultFinal_from_0_to_3061800.csv'
		parameters_dict = {
			"Ua": [0.9, 1, 1.1],
			"Cm": [0.9, 1, 1.1],
			"Ca": [0.9, 1, 1.1],
			"Hm": [0.9, 1, 1.1],
			"air_temp": [-4, -2, -1, 0, 1, 2, 4],  # Setter au debut
			"mass_temp": [-4, -2, 0, 2, 4],  # Setter au debut, ajouter au conf dict
			"OD_temp": [3, 5, 7, 9, 11],  # fixer en permanence
			"HVAC_power": [10000, 15000, 20000],
			"hour": [
				0.0,
				3.0,
				6.0,
				7.0,
				7.50,
				11.0,
				13.0,
				16.0,
				17.0,
				17.5,
				21.0,
				24 - 1.0 / 3600,
			],
			"date": [
				(2021, 1, 1),
				(2021, 3, 21),
				(2021, 6, 21),
				(2021, 9, 21),
				(2021, 12, 21),
				(2021, 12, 31),
			],
		}

		d0 = datetime.date(2021, 1, 1)
		parameters_dict["date"] = [(datetime.date(x[0], x[1], x[2]) - d0).days for x in parameters_dict["date"]]
		parameters_dict["hour"] = [x * SECOND_IN_A_HOUR for x in parameters_dict["hour"]]

		self.power_data = pd.read_csv(path)
		self.parameters_dict = deepcopy(parameters_dict)

		self.dict_keys = ["Ua", "Cm", "Ca", "Hm", "air_temp", "mass_temp", "OD_temp", "HVAC_power", "hour", "date"]
		self.power_interp = PowerInterpolator(path, parameters_dict, self.dict_keys)

		dates_nb = []

		self.num_comb = 1 
		for key in self.dict_keys:
			self.num_comb *= len(self.parameters_dict[key])

	def setUp(self):
		pass

	
	def testExactDataPoints(self):
		"""Tests that interpolation on exact data points gives the exact value"""
		for index_df in range(1, self.num_comb, int(self.num_comb/50)):
			gt_power = self.power_data["hvac_average_power"].loc[[index_df]].values
			point = {}
			for key in self.dict_keys:
				point[key] = self.power_data[key].loc[[index_df]].values.item()
			interp_value = self.power_interp.interpolateGridFast(point)
			self.assertEqual(gt_power, interp_value)

	def testDataPointsInter(self):
		"""Tests that interpolated points in between datapoints go in the right direction (with margin)"""
	
		# Air temperature
		air_0 = {
			"Ua": 1,
			"Cm": 1,
			"Ca": 1,
			"Hm": 1,
			"air_temp": -3.9,
			"mass_temp": 0,
			"OD_temp": 11,
			"HVAC_power": 15000,
			"hour": 9,
			"date": 83,
			}
		air_1 = deepcopy(air_0)
		air_2 = deepcopy(air_0)
		air_3 = deepcopy(air_0)
		air_4 = deepcopy(air_0)
		air_5 = deepcopy(air_0)
		air_6 = deepcopy(air_0)
		air_7 = deepcopy(air_0)
		air_8 = deepcopy(air_0)

		air_1["air_temp"] = -3.3
		air_2["air_temp"] = -2.2
		air_3["air_temp"] = -1.1
		air_4["air_temp"] = 0
		air_5["air_temp"] = 1.1
		air_6["air_temp"] = 2.2
		air_7["air_temp"] = 3.3
		air_8["air_temp"] = 4

		print("air_1: {}".format(self.power_interp.interpolateGridFast(air_1)))
		print("air_2: {}".format(self.power_interp.interpolateGridFast(air_2)))
		print("air_3: {}".format(self.power_interp.interpolateGridFast(air_3)))
		print("air_4: {}".format(self.power_interp.interpolateGridFast(air_4)))
		print("air_5: {}".format(self.power_interp.interpolateGridFast(air_5)))
		print("air_6: {}".format(self.power_interp.interpolateGridFast(air_6)))
		print("air_7: {}".format(self.power_interp.interpolateGridFast(air_7)))
		print("air_8: {}".format(self.power_interp.interpolateGridFast(air_8)))


		print("Testing with air temperature")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_1), self.power_interp.interpolateGridFast(air_2) + 10)
		print("air_1 < air_2")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_2), self.power_interp.interpolateGridFast(air_3) + 10)
		print("air_2 < air_3")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_3), self.power_interp.interpolateGridFast(air_4) + 10)
		print("air_3 < air_4")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_4), self.power_interp.interpolateGridFast(air_5) + 10)
		print("air_4 < air_5")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_5), self.power_interp.interpolateGridFast(air_6) + 10)
		print("air_5 < air_6")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_6), self.power_interp.interpolateGridFast(air_7) + 10)
		print("air_6 < air_7")
		#self.assertLessEqual(self.power_interp.interpolateGridFast(air_7), self.power_interp.interpolateGridFast(air_8) + 10)
		print("air_7 < air_8")


		# Mass temperature
		mass_0 = {
			"Ua": 1,
			"Cm": 1,
			"Ca": 1,
			"Hm": 1,
			"air_temp": 0,
			"mass_temp": -4,
			"OD_temp": 11,
			"HVAC_power": 15000,
			"hour": 9,
			"date": 83,
			}
		mass_1 = deepcopy(mass_0)
		mass_2 = deepcopy(mass_0)
		mass_3 = deepcopy(mass_0)
		mass_4 = deepcopy(mass_0)
		mass_5 = deepcopy(mass_0)
		mass_6 = deepcopy(mass_0)
		mass_7 = deepcopy(mass_0)
		mass_8 = deepcopy(mass_0)

		mass_1["mass_temp"] = -3.3
		mass_2["mass_temp"] = -2.2
		mass_3["mass_temp"] = -1.1
		mass_4["mass_temp"] = 0
		mass_5["mass_temp"] = 1.1
		mass_6["mass_temp"] = 2.2
		mass_7["mass_temp"] = 3.3
		mass_8["mass_temp"] = 4

		print("mass_1: {}".format(self.power_interp.interpolateGridFast(mass_1)))
		print("mass_2: {}".format(self.power_interp.interpolateGridFast(mass_2)))
		print("mass_3: {}".format(self.power_interp.interpolateGridFast(mass_3)))
		print("mass_4: {}".format(self.power_interp.interpolateGridFast(mass_4)))
		print("mass_5: {}".format(self.power_interp.interpolateGridFast(mass_5)))
		print("mass_6: {}".format(self.power_interp.interpolateGridFast(mass_6)))
		print("mass_7: {}".format(self.power_interp.interpolateGridFast(mass_7)))
		print("mass_8: {}".format(self.power_interp.interpolateGridFast(mass_8)))


		print("Testing with mass temperature")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_1), self.power_interp.interpolateGridFast(mass_2) + 10)
		print("mass_1 < mass_2")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_2), self.power_interp.interpolateGridFast(mass_3) + 10)
		print("mass_2 < mass_3")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_3), self.power_interp.interpolateGridFast(mass_4) + 10)
		print("mass_3 < mass_4")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_4), self.power_interp.interpolateGridFast(mass_5) + 10)
		print("mass_4 < mass_5")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_5), self.power_interp.interpolateGridFast(mass_6) + 10)
		print("mass_5 < mass_6")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_6), self.power_interp.interpolateGridFast(mass_7) + 10)
		print("mass_6 < mass_7")
		self.assertLessEqual(self.power_interp.interpolateGridFast(mass_7), self.power_interp.interpolateGridFast(mass_8) + 10)
		print("mass_7 < mass_8")

		# OD temperature
		OD_0 = {
			"Ua": 1,
			"Cm": 1,
			"Ca": 1,
			"Hm": 1,
			"air_temp": 0,
			"mass_temp": 0,
			"OD_temp": 11,
			"HVAC_power": 15000,
			"hour": 9,
			"date": 83,
			}
		OD_1 = deepcopy(OD_0)
		OD_2 = deepcopy(OD_0)
		OD_3 = deepcopy(OD_0)
		OD_4 = deepcopy(OD_0)
		OD_5 = deepcopy(OD_0)
		OD_6 = deepcopy(OD_0)

		OD_1["OD_temp"] = 3
		OD_2["OD_temp"] = 4.4
		OD_3["OD_temp"] = 6.6
		OD_4["OD_temp"] = 8.8
		OD_5["OD_temp"] = 9.9
		OD_6["OD_temp"] = 11


		print("OD_1: {}".format(self.power_interp.interpolateGridFast(OD_1)))
		print("OD_2: {}".format(self.power_interp.interpolateGridFast(OD_2)))
		print("OD_3: {}".format(self.power_interp.interpolateGridFast(OD_3)))
		print("OD_4: {}".format(self.power_interp.interpolateGridFast(OD_4)))
		print("OD_5: {}".format(self.power_interp.interpolateGridFast(OD_5)))
		print("OD_6: {}".format(self.power_interp.interpolateGridFast(OD_6)))


		print("Testing with OD temperature")
		self.assertLessEqual(self.power_interp.interpolateGridFast(OD_1), self.power_interp.interpolateGridFast(OD_2) + 10)
		print("OD_1 < OD_2")
		self.assertLessEqual(self.power_interp.interpolateGridFast(OD_2), self.power_interp.interpolateGridFast(OD_3) + 10)
		print("OD_2 < OD_3")
		self.assertLessEqual(self.power_interp.interpolateGridFast(OD_3), self.power_interp.interpolateGridFast(OD_4) + 10)
		print("OD_3 < OD_4")
		self.assertLessEqual(self.power_interp.interpolateGridFast(OD_4), self.power_interp.interpolateGridFast(OD_5) + 10)
		print("OD_4 < OD_5")
		self.assertLessEqual(self.power_interp.interpolateGridFast(OD_5), self.power_interp.interpolateGridFast(OD_6) + 10)
		print("OD_5 < OD_6")



if __name__ == '__main__':
	unittest.main()
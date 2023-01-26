import csv
import datetime
import json
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from interpolation import PowerInterpolator

SECOND_IN_A_HOUR = 3600


def point_to_index(point, parameters_dict, dict_keys):
    num_comb = 1
    num_comb_lower = {}
    for key in reversed(dict_keys):
        num_comb_lower[key] = num_comb
        num_comb *= len(parameters_dict[key])

    index = 0
    for key in dict_keys:
        ind_list = parameters_dict[key].index(point[key])
        index += ind_list * num_comb_lower[key]
    return index


def index_to_point(index, parameters_dict, dict_keys):
    num_comb = 1
    num_comb_lower = {}
    for key in reversed(dict_keys):
        num_comb_lower[key] = num_comb
        num_comb *= len(parameters_dict[key])

    point = {}
    div = 0
    for key in dict_keys:
        key_id = int((index - div) / num_comb_lower[key])
        point[key] = parameters_dict[key][key_id]
        div += key_id * num_comb_lower[key]
    return point


class TestPowerInterpolator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPowerInterpolator, self).__init__(*args, **kwargs)

        path = "./mergedGridSearchResultFinal.npy"
        with open("./interp_parameters_dict.json") as json_file:
            parameters_dict = json.load(json_file)
        with open("./interp_dict_keys.csv") as f:
            reader = csv.reader(f)
            self.dict_keys = list(reader)[0]

        self.power_data = np.load(path)
        self.parameters_dict = deepcopy(parameters_dict)

        self.power_interp = PowerInterpolator(path, parameters_dict, self.dict_keys)

        dates_nb = []

        self.num_comb = 1
        self.num_comb_lower = {}
        for key in reversed(self.dict_keys):
            self.num_comb_lower[key] = self.num_comb
            self.num_comb *= len(self.parameters_dict[key])

    def setUp(self):
        pass

    def testExactDataPoints(self):
        """Tests that interpolation on exact data points gives the exact value"""
        for index_df in range(0, self.num_comb, int(self.num_comb / 50)):
            gt_power = self.power_data[index_df]
            point = index_to_point(index_df, self.parameters_dict, self.dict_keys)
            interp_value = self.power_interp.interpolateGridFast(point)

            self.assertEqual(gt_power, interp_value)

        air_8 = {
            "Ua_ratio": 1,
            "Cm_ratio": 1,
            "Ca_ratio": 1,
            "Hm_ratio": 1,
            "air_temp": 4,
            "mass_temp": 0,
            "OD_temp": 11,
            "HVAC_power": 15000,
            "hour": 11.0 * SECOND_IN_A_HOUR,
            "date": 79,
        }
        index = point_to_index(air_8, self.parameters_dict, self.dict_keys)
        gt_power = self.power_data[index_df]
        interp_value = self.power_interp.interpolateGridFast(point)
        self.assertEqual(gt_power, interp_value)

    def testIndexPoint(self):
        air_0 = {
            "Ua_ratio": 1,
            "Cm_ratio": 1,
            "Ca_ratio": 1,
            "Hm_ratio": 1,
            "air_temp": -1,
            "mass_temp": 0,
            "OD_temp": 11,
            "HVAC_power": 15000,
            "hour": 11.0 * SECOND_IN_A_HOUR,
            "date": 79,
        }
        index = point_to_index(air_0, self.parameters_dict, self.dict_keys)
        point = index_to_point(index, self.parameters_dict, self.dict_keys)
        self.assertEqual(air_0, point)

    def testDataPointsInter(self):
        """Tests that interpolated points in between datapoints go in the right direction (with margin)"""

        # Air temperature
        air_0 = {
            "Ua_ratio": 1,
            "Cm_ratio": 1,
            "Ca_ratio": 1,
            "Hm_ratio": 1,
            "air_temp": -1,
            "mass_temp": 0,
            "OD_temp": 9,
            "HVAC_power": 15000,
            "hour": 11.0 * SECOND_IN_A_HOUR,
            "date": 79,
        }
        air_1 = deepcopy(air_0)
        air_2 = deepcopy(air_0)
        air_3 = deepcopy(air_0)
        air_4 = deepcopy(air_0)
        air_5 = deepcopy(air_0)
        air_6 = deepcopy(air_0)
        air_7 = deepcopy(air_0)
        air_8 = deepcopy(air_0)

        air_1["air_temp"] = -4
        air_2["air_temp"] = -3
        air_3["air_temp"] = -1
        air_4["air_temp"] = 0
        air_5["air_temp"] = 1
        air_6["air_temp"] = 2
        air_7["air_temp"] = 3
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
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_1),
            self.power_interp.interpolateGridFast(air_2),
        )
        print("air_1 < air_2")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_2),
            self.power_interp.interpolateGridFast(air_3),
        )
        print("air_2 < air_3")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_3),
            self.power_interp.interpolateGridFast(air_4),
        )
        print("air_3 < air_4")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_4),
            self.power_interp.interpolateGridFast(air_5),
        )
        print("air_4 < air_5")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_5),
            self.power_interp.interpolateGridFast(air_6),
        )
        print("air_5 < air_6")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_6),
            self.power_interp.interpolateGridFast(air_7),
        )
        print("air_6 < air_7")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(air_7),
            self.power_interp.interpolateGridFast(air_8),
        )
        print("air_7 < air_8")

        # Mass temperature
        mass_0 = {
            "Ua_ratio": 1,
            "Cm_ratio": 1,
            "Ca_ratio": 1,
            "Hm_ratio": 1,
            "air_temp": 4,
            "mass_temp": 4,
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
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_1),
            self.power_interp.interpolateGridFast(mass_2) + 10,
        )
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_2),
            self.power_interp.interpolateGridFast(mass_3) + 10,
        )
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_3),
            self.power_interp.interpolateGridFast(mass_4) + 10,
        )
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_4),
            self.power_interp.interpolateGridFast(mass_5) + 10,
        )
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_5),
            self.power_interp.interpolateGridFast(mass_6) + 10,
        )
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_6),
            self.power_interp.interpolateGridFast(mass_7) + 10,
        )
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(mass_7),
            self.power_interp.interpolateGridFast(mass_8) + 10,
        )

        # OD temperature
        OD_0 = {
            "Ua_ratio": 1,
            "Cm_ratio": 1,
            "Ca_ratio": 1,
            "Hm_ratio": 1,
            "air_temp": 4,
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
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(OD_1),
            self.power_interp.interpolateGridFast(OD_2) + 10,
        )
        print("OD_1 < OD_2")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(OD_2),
            self.power_interp.interpolateGridFast(OD_3) + 10,
        )
        print("OD_2 < OD_3")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(OD_3),
            self.power_interp.interpolateGridFast(OD_4) + 10,
        )
        print("OD_3 < OD_4")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(OD_4),
            self.power_interp.interpolateGridFast(OD_5) + 10,
        )
        print("OD_4 < OD_5")
        self.assertLessEqual(
            self.power_interp.interpolateGridFast(OD_5),
            self.power_interp.interpolateGridFast(OD_6) + 10,
        )
        print("OD_5 < OD_6")

        ### HVAC power
        hvacp_0 = {
            "Ua_ratio": 1,
            "Cm_ratio": 1,
            "Ca_ratio": 1,
            "Hm_ratio": 1,
            "air_temp": 6,
            "mass_temp": 0,
            "OD_temp": 9,
            "HVAC_power": 10000,
            "hour": 11.0 * SECOND_IN_A_HOUR,
            "date": 79,
        }

        hvacp_1 = deepcopy(hvacp_0)
        hvacp_2 = deepcopy(hvacp_0)
        hvacp_3 = deepcopy(hvacp_0)
        hvacp_4 = deepcopy(hvacp_0)
        hvacp_5 = deepcopy(hvacp_0)
        hvacp_6 = deepcopy(hvacp_0)

        hvacp_1["HVAC_power"] = 10000
        hvacp_2["HVAC_power"] = 11000
        hvacp_3["HVAC_power"] = 13000
        hvacp_4["HVAC_power"] = 16000
        hvacp_5["HVAC_power"] = 18000
        hvacp_6["HVAC_power"] = 20000

        print("hvacp_0: {}".format(self.power_interp.interpolateGridFast(hvacp_0)))
        print("hvacp_1: {}".format(self.power_interp.interpolateGridFast(hvacp_1)))
        print("hvacp_2: {}".format(self.power_interp.interpolateGridFast(hvacp_2)))
        print("hvacp_3: {}".format(self.power_interp.interpolateGridFast(hvacp_3)))
        print("hvacp_4: {}".format(self.power_interp.interpolateGridFast(hvacp_4)))
        print("hvacp_5: {}".format(self.power_interp.interpolateGridFast(hvacp_5)))
        print("hvacp_6: {}".format(self.power_interp.interpolateGridFast(hvacp_6)))


if __name__ == "__main__":
    unittest.main()

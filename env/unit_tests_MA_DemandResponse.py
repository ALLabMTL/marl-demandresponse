from MA_DemandResponse import MADemandResponseEnv, HVAC, SingleHouse, ClusterHouses, PowerGrid
import unittest
import gym
import ray
import numpy as np
import warnings
import random
from copy import deepcopy
from datetime import datetime, timedelta, time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List, Any
import sys
sys.path.insert(1, "../marl-demandresponse")
sys.path.append("..")

from utils import applyPropertyNoise


class TestHVAC(unittest.TestCase):

	def setUp(self):
		hvac_properties = {	
			"id": 1,
			"COP": 2.5,
			"cooling_capacity": 15000,
			"latent_cooling_fraction": 0.35,
			"lockout_duration": 12,
		}

		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		self.hvac = HVAC(hvac_properties, time_step)
	
	def testConsumptionQ(self):
		"""Tests consumption and heat production"""
		self.hvac.turned_on = False
		self.assertEqual(self.hvac.get_Q(), 0)
		self.assertEqual(self.hvac.power_consumption(), 0)

		self.hvac.turned_on = True
		self.assertEqual(self.hvac.get_Q(), -15000/(1 + 0.35))
		self.assertEqual(self.hvac.power_consumption(), 15000/2.5)

	def testLockout(self):
		"""Tests lockout"""
		self.hvac.turned_on = True
		self.assertFalse(self.hvac.lockout)

		self.hvac.step(True)
		self.assertTrue(self.hvac.turned_on)
		self.assertFalse(self.hvac.lockout)

		self.hvac.step(False)
		self.assertFalse(self.hvac.turned_on)
		self.assertTrue(self.hvac.lockout)

		self.hvac.step(True)
		self.assertFalse(self.hvac.turned_on)
		self.assertTrue(self.hvac.lockout)
		self.assertEqual(self.hvac.seconds_since_off, 4)

		self.hvac.step(True)
		self.assertFalse(self.hvac.turned_on)
		self.assertTrue(self.hvac.lockout)
		self.assertEqual(self.hvac.seconds_since_off, 8)

		self.hvac.step(True)
		self.assertTrue(self.hvac.turned_on)
		self.assertFalse(self.hvac.lockout)
		self.assertEqual(self.hvac.seconds_since_off, 0)

		self.hvac.step(True)
		self.assertTrue(self.hvac.turned_on)
		self.assertFalse(self.hvac.lockout)
		self.assertEqual(self.hvac.seconds_since_off, 0)


class TestHouse(unittest.TestCase):

	def setUp(self):
		hvac_properties = {	
			"id": 'a1',
			"COP": 2.5,
			"cooling_capacity": 15000,
			"latent_cooling_fraction": 0.35,
			"lockout_duration": 12,
		}
		self.houses_properties = {
			"id": 1,
			"init_air_temp": 20,
			"init_mass_temp": 20,
			"target_temp": 20,
			"deadband": 2,
			"Ua" : 2.18e02,
			"Cm" : 3.45e06,
			"Ca" : 9.08e05,
			"Hm" : 2.84e03,
			"window_area" : 7.175,
			"shading_coeff": 0.67 
		}
		self.houses_properties["hvac_properties"] = [hvac_properties]

		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		self.house = SingleHouse(self.houses_properties, time_step)
		self.house2 = SingleHouse(self.houses_properties, time_step)

		self.house.hvacs["a1"].turned_on = False	
		self.house2.hvacs["a1"].turned_on = False	

	def testSolarGainTime(self):

		# Test time of day
		datetime_0 = datetime(2021, 6, 15, 0, 0, 0, 0)
		datetime_1 = datetime(2021, 6, 15, 7, 29, 0, 0)
		datetime_2 = datetime(2021, 6, 15, 17, 31, 0, 0)
		datetime_3 = datetime(2021, 6, 15, 12, 0, 0, 0)
		datetime_4 = datetime(2021, 6, 15, 7, 31, 0, 0)
		datetime_5 = datetime(2021, 6, 15, 17, 29, 0, 0)

		self.assertEqual(self.house.house_solar_gain(datetime_0), 0)
		self.assertEqual(self.house.house_solar_gain(datetime_1), 0)
		self.assertEqual(self.house.house_solar_gain(datetime_2), 0)
		self.assertGreater(self.house.house_solar_gain(datetime_3), 0)
		self.assertGreater(self.house.house_solar_gain(datetime_4), 0)
		self.assertGreater(self.house.house_solar_gain(datetime_5), 0)

	def testSolarGainDate(self):
		# Test date of year (To be remade)
		datetime_6_1 = datetime(2021, 12, 21, 8, 0, 0, 0)
		datetime_6_2 = datetime(2021, 12, 21, 10, 15, 0, 0)
		datetime_6_3 = datetime(2021, 12, 21, 12, 30, 0, 0)
		datetime_6_4 = datetime(2021, 12, 21, 14, 45, 0, 0)
		datetime_6_5 = datetime(2021, 12, 21, 17, 0, 0, 0)

		datetime_7_1 = datetime(2021, 3, 21, 8, 0, 0, 0)
		datetime_7_2 = datetime(2021, 3, 21, 10, 15, 0, 0)
		datetime_7_3 = datetime(2021, 3, 21, 12, 30, 0, 0)
		datetime_7_4 = datetime(2021, 3, 21, 14, 45, 0, 0)
		datetime_7_5 = datetime(2021, 3, 21, 17, 0, 0, 0)

		datetime_8_1 = datetime(2021, 6, 21, 8, 0, 0, 0)
		datetime_8_2 = datetime(2021, 6, 21, 10, 15, 0, 0)
		datetime_8_3 = datetime(2021, 6, 21, 12, 30, 0, 0)
		datetime_8_4 = datetime(2021, 6, 21, 14, 45, 0, 0)
		datetime_8_5 = datetime(2021, 6, 21, 17, 0, 0, 0)

		datetime_9_1 = datetime(2021, 9, 21, 8, 0, 0, 0)
		datetime_9_2 = datetime(2021, 9, 21, 10, 15, 0, 0)
		datetime_9_3 = datetime(2021, 9, 21, 12, 30, 0, 0)
		datetime_9_4 = datetime(2021, 9, 21, 14, 45, 0, 0)
		datetime_9_5 = datetime(2021, 9, 21, 17, 0, 0, 0)

		sum_6 = self.house.house_solar_gain(datetime_6_1) + self.house.house_solar_gain(datetime_6_2) + self.house.house_solar_gain(datetime_6_3) + self.house.house_solar_gain(datetime_6_4)  + self.house.house_solar_gain(datetime_6_5)
		sum_7 = self.house.house_solar_gain(datetime_7_1) + self.house.house_solar_gain(datetime_7_2) + self.house.house_solar_gain(datetime_7_3) + self.house.house_solar_gain(datetime_7_4)  + self.house.house_solar_gain(datetime_7_5)
		sum_8 = self.house.house_solar_gain(datetime_8_1) + self.house.house_solar_gain(datetime_8_2) + self.house.house_solar_gain(datetime_8_3) + self.house.house_solar_gain(datetime_8_4)  + self.house.house_solar_gain(datetime_8_5)
		sum_9 = self.house.house_solar_gain(datetime_9_1) + self.house.house_solar_gain(datetime_9_2) + self.house.house_solar_gain(datetime_9_3) + self.house.house_solar_gain(datetime_9_4)  + self.house.house_solar_gain(datetime_9_5)

		print("Dec: {}".format(sum_6))
		print("March: {}".format(sum_7))
		print("June: {}".format(sum_8))
		print("Sept: {}".format(sum_9))

		#self.assertGreater(sum_8, sum_9)  # June more sun than Sept.
		#self.assertGreater(sum_8, sum_7)  # June more sun than March
		#self.assertGreater(sum_7, sum_6)  # March more sun than Dec
		#self.assertGreater(sum_9, sum_6)  # Sept more sun than Dec

	def testSolarGainDivide(self):
		datetime_3 = datetime(2021, 6, 15, 12, 0, 0, 0)

		ori = self.house.house_solar_gain(datetime_3)
		self.house.window_area *= 0.5
		self.house.shading_coeff *= 0.5
		mod = self.house.house_solar_gain(datetime_3)

		self.assertEqual(ori, mod * 4)

	def testInitMassTemperature(self):
		"""Test that a higher initial mass temperature heats the house more"""

		date_time = datetime(2021, 6, 15, 0, 0, 0, 0)
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		od_temp = 25

		self.house.current_mass_temp = 30
		self.house2.current_mass_temp = 22

		self.house.step(od_temp, time_step, date_time)
		self.house2.step(od_temp, time_step, date_time)
		date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

	def testODTemperature(self):
		"""Test that a higher outdoors temperature heats the house more"""
		date_time = datetime(2021, 6, 15, 0, 0, 0, 0)
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		start_temp = self.house.current_temp

		od_temp_1 = 30
		od_temp_2 = 22


		self.house.step(od_temp_1, time_step, date_time)
		self.house2.step(od_temp_2, time_step, date_time)
		date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp_1, time_step, date_time)
			self.house2.step(od_temp_2, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp_1, time_step, date_time)
			self.house2.step(od_temp_2, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		self.assertGreater(self.house2.current_temp, start_temp)

	def testUa(self):
		"""Test that with a smaller Ua (wall conductance) there is less heat coming from outside"""
		date_time = datetime(2021, 6, 15, 0, 0, 0, 0)
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		start_temp = self.house.current_temp

		od_temp = 30

		self.house.Ua = 2.18e02
		self.house2.Ua = 2.18e02/2

		self.house.step(od_temp, time_step, date_time)
		self.house2.step(od_temp, time_step, date_time)
		date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)	

		self.assertGreater(self.house2.current_temp, start_temp)

	def testCa(self):
		"""Test that with a smaller Ca (air mass) the heat coming from outside warms the air and house mass faster"""
		date_time = datetime(2021, 6, 15, 0, 0, 0, 0)
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		start_temp = self.house.current_temp

		od_temp = 30

		self.house.Ca = 9.08e05
		self.house2.Ca = 9.08e05*2

		self.house.step(od_temp, time_step, date_time)
		self.house2.step(od_temp, time_step, date_time)
		date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)	

		self.assertGreater(self.house2.current_temp, start_temp)

	def testCm(self):
		"""Test that with a smaller Cm (house thermal mass) the heat coming from outside warms the air and house mass faster"""
		date_time = datetime(2021, 6, 15, 0, 0, 0, 0)
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		start_temp = self.house.current_temp

		od_temp = 30

		self.house.Cm = 3.45e06
		self.house2.Cm = 3.45e06*2

		self.house.step(od_temp, time_step, date_time)
		self.house2.step(od_temp, time_step, date_time)
		date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)	

		self.assertGreater(self.house2.current_temp, start_temp)

	def testHm(self):
		"""Test that with a higher Hm (house thermal mass), a high mass temp vs air mass warms the air faster but cools mass faster"""
		date_time = datetime(2021, 6, 15, 0, 0, 0, 0)
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

		start_temp = self.house.current_temp

		od_temp = 25

		self.house.Hm = 2.84e03
		self.house2.Hm = 2.84e03/2

		self.house.current_mass_temp = 30
		self.house2.current_mass_temp = 30


		self.house.step(od_temp, time_step, date_time)
		self.house2.step(od_temp, time_step, date_time)
		date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertLess(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertLess(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time)
			self.house2.step(od_temp, time_step, date_time)
			date_time += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertLess(self.house.current_mass_temp, self.house2.current_mass_temp)	

		self.assertGreater(self.house2.current_temp, start_temp)

	def testSunEffect(self):
		"""Test that with Sun the temperature warms faster"""
		date_time_1 = datetime(2021, 6, 15, 12, 0, 0, 0) # Midday
		date_time_2 = datetime(2021, 6, 15, 0, 0, 0, 0) # Midnight
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)		

		start_temp = self.house.current_temp

		od_temp = 25

		self.house.step(od_temp, time_step, date_time_1)
		self.house2.step(od_temp, time_step, date_time_2)
		date_time_1 += time_step
		date_time_2 += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time_1)
			self.house2.step(od_temp, time_step, date_time_2)
			date_time_1 += time_step
			date_time_2 += time_step
		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time_1)
			self.house2.step(od_temp, time_step, date_time_2)
			date_time_1 += time_step
			date_time_2 += time_step

		self.assertGreater(self.house.current_temp, self.house2.current_temp)
		self.assertGreater(self.house.current_mass_temp, self.house2.current_mass_temp)	

		self.assertGreater(self.house2.current_temp, start_temp)

	def testHVACEffect(self):
		"""Test that when HVAC is on, temperature goes down."""
		date_time_1 = datetime(2021, 6, 15, 12, 0, 0, 0) # Midday
		time_step = timedelta(days=0, seconds=4, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)	

		self.house.hvacs["a1"].turned_on = True	

		start_temp = self.house.current_temp

		od_temp = 30

		self.house.step(od_temp, time_step, date_time_1)
		date_time_1 += time_step

		self.assertGreater(start_temp, self.house.current_temp)
		self.assertGreater(start_temp, self.house.current_mass_temp)

		for i in range(49):
			self.house.step(od_temp, time_step, date_time_1)
			date_time_1 += time_step
		self.assertGreater(start_temp, self.house.current_temp)
		self.assertGreater(start_temp, self.house.current_mass_temp)

		for i in range(950):
			self.house.step(od_temp, time_step, date_time_1)
			date_time_1 += time_step

		self.assertGreater(start_temp, self.house.current_temp)
		self.assertGreater(start_temp, self.house.current_mass_temp)

if __name__ == '__main__':
	unittest.main()



config_dict = {

# House properties
# The house is modelled as a 10mx10m square with 2.5m height, 8 windows of 1.125 m² and 2 doors of 2m² each.
# The formulas for Ua, Cm, Ca and Hm are mainly taken here: http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide

"default_house_prop" : {
	"id": 1,
	"init_temp": 20,
	"target_temp": 20,
	"deadband": 2,
	"Ua" : 2.18e02,								# House walls conductance (W/K) (75 for walls and ceiling, 4.5 for two doors, 58 for windows). Multiplied by 3 to account for drafts (according to https://dothemath.ucsd.edu/2012/11/this-thermal-house/)
	"Cm" : 3.45e06,							# House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)
	"Ca" : 9.08e05,						# Air thermal mass in the house (J/K): 3 * (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)
	"Hm" : 2.84e03,							# House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.14 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))
	"window_area" : 7.175, 						# Gross window area, in m^2  
	"shading_coeff": 0.67 					# Window Solar Heat Gain Coefficient, look-up table in Gridlab reference
},

"noise_house_prop" : {
	"std_start_temp": 3,		# Std noise on starting temperature
	"std_target_temp": 1,     # Std Noise on target temperature
	"factor_thermo_low": 0.9,   # Lowest random factor for Ua, Cm, Ca, Hm
	"factor_thermo_high": 1.1,   # Highest random factor for Ua, Cm, Ca, Hm
},

"noise_house_prop_test" : {
	"std_start_temp": 1,		# Std noise on starting temperature
	"std_target_temp": 1,     # Std Noise on target temperature
	"factor_thermo_low": 0.9,   # Lowest random factor for Ua, Cm, Ca, Hm
	"factor_thermo_high": 1.1,   # Highest random factor for Ua, Cm, Ca, Hm
},

# HVAC properties

"default_hvac_prop" : {
	"id": 1,
	"COP": 2.5,									# Coefficient of performance (power spent vs heat displaced)
	"cooling_capacity": 15000,					# Cooling capacity (W)
	"latent_cooling_fraction": 0.35,			# Fraction of latent cooling w.r.t. sensible cooling
	"lockout_duration": 40						# In seconds
},

"noise_hvac_prop" : {
	"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
	"factor_COP_low": 0.9,   # Lowest random factor for COP
	"factor_COP_high": 1.1,   # Highest random factor for COP
	"factor_cooling_capacity_low": 0.9,   # Lowest random factor for cooling_capacity
	"factor_cooling_capacity_high": 1.1,   # Highest random factor for cooling_capacity
},

"noise_hvac_prop_test" : {
	"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
	"factor_COP_low": 0.9,   # Lowest random factor for COP
	"factor_COP_high": 1.1,   # Highest random factor for COP
	"factor_cooling_capacity_low": 0.9,   # Lowest random factor for cooling_capacity
	"factor_cooling_capacity_high": 1.1,   # Highest random factor for cooling_capacity
},


# Env properties

"default_env_properties" : {
	"base_datetime": '2021-01-01 00:00:00',   	# Start date and time (Y-m-d H:M:S)
	"time_step": 4,							# Time step in seconds
	"cluster_properties": {
		"day_temp": 26.5,							# Day temperature
		"night_temp": 26.5,						# Night temperature
		"temp_std": 0.5,						# Noise std dev on the temperature
		"nb_agents": 1,							# Number of agents (or houses)
	},
	"power_grid_properties": {
		"avg_power_per_hvac": 910,					# Per hvac. In Watts. Based on average necessary power for bang-bang controller.
		"init_signal_per_hvac": 910, 				# Per hvac.
		"signal_mode": "regular_steps",					# Mode of the signal. Currently available: none, sinusoidal, regular_steps
		"signal_parameters": {
			"none": {},
			"sinusoidals": {
				"periods": [400, 1200],					# In seconds
				"amplitude_ratios": [0.1, 0.3],			# As a ratio of avg_power_per_hvac
			},
			"regular_steps": {
				"periods": [300],					# In seconds
				"ratios": [0.7],					# Ratio of time "on"
			}
		}
	},
	"alpha": 2e-5,									# Tradeoff parameter for loss function: temperature penalty + alpha * regulation signal penalty
},

}

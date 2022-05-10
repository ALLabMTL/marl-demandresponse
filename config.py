
config_dict = {

# House properties
# The house is modelled as a 10mx10m square with 2.5m height, 8 windows of 1.125 m² and 2 doors of 2m² each.
# The formulas for Ua, Cm, Ca and Hm are mainly taken here: http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide

"default_house_prop" : {
	"id": 1,
	"init_air_temp": 20,
	"init_mass_temp": 20,
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
	"noise_mode": "small_noise",	#Can be: no_noise, small_noise, big_noise, small_start_temp, big_start_temp
	"noise_parameters": {
		"no_noise": {
			"std_start_temp": 0,		# Std noise on starting temperature
			"std_target_temp": 0,     # Std Noise on target temperature
			"factor_thermo_low": 1,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"small_noise": {
			"std_start_temp": 3,		# Std noise on starting temperature
			"std_target_temp": 1,     # Std Noise on target temperature
			"factor_thermo_low": 0.9,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1.1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"big_noise": {
			"std_start_temp": 5,		# Std noise on starting temperature
			"std_target_temp": 2,     # Std Noise on target temperature
			"factor_thermo_low": 0.8,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1.2,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"small_start_temp": {
			"std_start_temp": 3,		# Std noise on starting temperature
			"std_target_temp": 0,     # Std Noise on target temperature
			"factor_thermo_low": 1,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"big_start_temp": {
			"std_start_temp": 5,		# Std noise on starting temperature
			"std_target_temp": 0,     # Std Noise on target temperature
			"factor_thermo_low": 1,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
	},
},

"noise_house_prop_test" : {
	"noise_mode": "small_noise",	#Can be: no_noise, small_noise, big_noise, small_start_temp, big_start_temp
	"noise_parameters": {
		"no_noise": {
			"std_start_temp": 0,		# Std noise on starting temperature
			"std_target_temp": 0,     # Std Noise on target temperature
			"factor_thermo_low": 1,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"small_noise": {
			"std_start_temp": 3,		# Std noise on starting temperature
			"std_target_temp": 1,     # Std Noise on target temperature
			"factor_thermo_low": 0.9,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1.1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"big_noise": {
			"std_start_temp": 5,		# Std noise on starting temperature
			"std_target_temp": 2,     # Std Noise on target temperature
			"factor_thermo_low": 0.8,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1.2,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"small_start_temp": {
			"std_start_temp": 3,		# Std noise on starting temperature
			"std_target_temp": 0,     # Std Noise on target temperature
			"factor_thermo_low": 1,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
		"big_start_temp": {
			"std_start_temp": 5,		# Std noise on starting temperature
			"std_target_temp": 0,     # Std Noise on target temperature
			"factor_thermo_low": 1,   # Lowest random factor for Ua, Cm, Ca, Hm
			"factor_thermo_high": 1,   # Highest random factor for Ua, Cm, Ca, Hm
		},
	},
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
	"noise_mode": "small_noise",	#Can be: no_noise, small_noise, big_noise
	"noise_parameters": {
		"no_noise": {
			"std_latent_cooling_fraction": 0,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 1,   # Lowest random factor for COP
			"factor_COP_high": 1,   # Highest random factor for COP
			"factor_cooling_capacity_low": 1,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1,   # Highest random factor for cooling_capacity
		},
		"small_noise": {
			"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 0.95,   # Lowest random factor for COP
			"factor_COP_high": 1.05,   # Highest random factor for COP
			"factor_cooling_capacity_low": 0.95,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1.05,   # Highest random factor for cooling_capacity
		},
		"big_noise": {
			"std_latent_cooling_fraction": 0.1,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 0.85,   # Lowest random factor for COP
			"factor_COP_high": 1.15,   # Highest random factor for COP
			"factor_cooling_capacity_low": 0.85,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1.15,   # Highest random factor for cooling_capacity
		},
	},
},

"noise_hvac_prop_test" : {
	"noise_mode": "small_noise",	#Can be: no_noise, small_noise, big_noise
	"noise_parameters": {
		"no_noise": {
			"std_latent_cooling_fraction": 0,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 1,   # Lowest random factor for COP
			"factor_COP_high": 1,   # Highest random factor for COP
			"factor_cooling_capacity_low": 1,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1,   # Highest random factor for cooling_capacity
		},
		"small_noise": {
			"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 0.95,   # Lowest random factor for COP
			"factor_COP_high": 1.05,   # Highest random factor for COP
			"factor_cooling_capacity_low": 0.95,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1.05,   # Highest random factor for cooling_capacity
		},
		"big_noise": {
			"std_latent_cooling_fraction": 0.1,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 0.85,   # Lowest random factor for COP
			"factor_COP_high": 1.15,   # Highest random factor for COP
			"factor_cooling_capacity_low": 0.85,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1.15,   # Highest random factor for cooling_capacity
		},
	},
},



# Env properties

"default_env_prop" : {
	"base_datetime": '2021-01-01 00:00:00',   	# Start date and time (Y-m-d H:M:S)
	"time_step": 4,							# Time step in seconds
	"cluster_prop": {
		"temp_mode": "noisy_sinusoidal",			# Can be: constant, sinusoidal, noisy_sinusoidal
		"temp_parameters": {
			"constant": {
				"day_temp": 26.5,				# Day temperature
				"night_temp": 26.5,				# Night temperature
				"temp_std": 0,					# Noise std dev on the temperature
			},
			"sinusoidal": {
				"day_temp": 30,
				"night_temp": 23,
				"temp_std": 0,
			},
			"noisy_sinusoidal": {
				"day_temp": 30,
				"night_temp": 23,
				"temp_std": 0.5,
			},

		},
		"nb_agents": 1,							# Number of agents (or houses)
	},
	"power_grid_prop": {
		"avg_power_per_hvac": 4200,					# Per hvac. In Watts. Based on average necessary power for bang-bang controller.
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
	"alpha_temp": 1,									# Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
	"alpha_sig": 1,									# Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
},

# NN properties

"nn_prop": {
    "layers": [20,20],
    "gamma": 0.99,
    "tau": 0.01,
    "buffer_capacity": 10000,
    "lr": 1e-3,
    },

}

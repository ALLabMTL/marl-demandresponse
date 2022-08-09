
config_dict = {

# House properties
# The house is modelled as a 10mx10m square with 2.5m height, 8 windows of 1.125 m² and 2 doors of 2m² each.
# The formulas for Ua, Cm, Ca and Hm are mainly taken here: http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide

"default_house_prop" : {
	"id": 1,
	"init_air_temp": 20,
	"init_mass_temp": 20,
	"target_temp": 20,
	"deadband": 0,
	"Ua" : 2.18e02,								# House walls conductance (W/K) (75 for walls and ceiling, 4.5 for two doors, 58 for windows). Multiplied by 3 to account for drafts (according to https://dothemath.ucsd.edu/2012/11/this-thermal-house/)
	"Cm" : 3.45e06,							# House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)
	"Ca" : 9.08e05,						# Air thermal mass in the house (J/K): 3 * (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)
	"Hm" : 2.84e03,							# House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.14 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))
	"window_area" : 7.175, 						# Gross window area, in m^2  
	"shading_coeff": 0.67, 					# Window Solar Heat Gain Coefficient, look-up table in Gridlab reference
	"solar_gain_bool": True,						# Boolean to model the solar gain
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
			"cooling_capacity_list": {
				10000: [10000],
				15000: [15000]
			}
			#"std_latent_cooling_fraction": 0,     # Std Gaussian noise on latent_cooling_fraction
			#"factor_COP_low": 1,   # Lowest random factor for COP
			#"factor_COP_high": 1,   # Highest random factor for COP
			#"factor_cooling_capacity_low": 1,   # Lowest random factor for cooling_capacity
			#"factor_cooling_capacity_high": 1,   # Highest random factor for cooling_capacity
		},
		"small_noise": {
			"cooling_capacity_list": {
				10000: [7500, 10000],
				15000: [10000, 15000]
			}

			#"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
			#"factor_COP_low": 0.95,   # Lowest random factor for COP
			#"factor_COP_high": 1.05,   # Highest random factor for COP
			#"factor_cooling_capacity_low": 0.9,   # Lowest random factor for cooling_capacity
			#"factor_cooling_capacity_high": 1.1,   # Highest random factor for cooling_capacity
		},
		"big_noise": {
			"cooling_capacity_list": {
				10000: [7500, 10000, 12500],
				15000: [10000, 15000, 20000]
			}
			#"std_latent_cooling_fraction": 0.1,     # Std Gaussian noise on latent_cooling_fraction
			#"factor_COP_low": 0.85,   # Lowest random factor for COP
			#"factor_COP_high": 1.15,   # Highest random factor for COP
			#"factor_cooling_capacity_low": 0.6666667,   # Lowest random factor for cooling_capacity
			#"factor_cooling_capacity_high": 1.3333333333,   # Highest random factor for cooling_capacity
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
			"factor_cooling_capacity_low": 0.9,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1.1,   # Highest random factor for cooling_capacity
		},
		"big_noise": {
			"std_latent_cooling_fraction": 0.1,     # Std Gaussian noise on latent_cooling_fraction
			"factor_COP_low": 0.85,   # Lowest random factor for COP
			"factor_COP_high": 1.15,   # Highest random factor for COP
			"factor_cooling_capacity_low": 0.6666667,   # Lowest random factor for cooling_capacity
			"factor_cooling_capacity_high": 1.3333333333,   # Highest random factor for cooling_capacity
		},
	},
},



# Env properties

"default_env_prop" : {
	"start_datetime": '2021-01-01 00:00:00',   	# Start date and time (Y-m-d H:M:S)
	"start_datetime_mode" : "random",		# Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)
	"time_step": 4,							# Time step in seconds
	"cluster_prop": {
		"temp_mode": "noisy_sinusoidal",			# Can be: constant, sinusoidal, noisy_sinusoidal
		"temp_parameters": {
			"constant": {
				"day_temp": 26.5,				# Day temperature
				"night_temp": 26.5,				# Night temperature
				"temp_std": 0,					# Noise std dev on the temperature
				"random_phase_offset": False,
			},
			"sinusoidal": {
				"day_temp": 30,
				"night_temp": 23,
				"temp_std": 0,
				"random_phase_offset": False,
			},
			"sinusoidal_hot": {
				"day_temp": 30,
				"night_temp": 28,
				"temp_std": 0,
				"random_phase_offset": False,
			},
			"sinusoidal_heatwave": {
				"day_temp": 34,
				"night_temp": 28,
				"temp_std": 0,			
				"random_phase_offset": False,
			},
			"sinusoidal_cold": {
				"day_temp": 24,
				"night_temp": 22,
				"temp_std": 0,
				"random_phase_offset": False,
			},
			"noisy_sinusoidal": {
				"day_temp": 30,
				"night_temp": 23,
				"temp_std": 0.5,
				"random_phase_offset": False,
			},
			"noisy_sinusoidal_hot": {
				"day_temp": 30,
				"night_temp": 28,
				"temp_std": 0.5,			
				"random_phase_offset": False,
			},
			"noisy_sinusoidal_heatwave": {
				"day_temp": 34,
				"night_temp": 28,
				"temp_std": 0.5,			
				"random_phase_offset": False,
			},
			"noisy_sinusoidal_cold": {
				"day_temp": 24,
				"night_temp": 22,
				"temp_std": 0.5,		
				"random_phase_offset": False,	
			},			
			"shifting_sinusoidal": {
				"day_temp": 30,
				"night_temp": 23,
				"temp_std": 0,
				"random_phase_offset": True,
			}
		},
		"nb_agents": 1,							# Number of houses
		"nb_agents_comm": 10,					# Maximal number of houses a single house communicates with
		"agents_comm_mode": "neighbours",		# Communication mode
		"agents_comm_parameters": {
			"neighbours_2D": {
				"row_size": 5,					# Row side length
				"distance_comm": 2,					# Max distance between two communicating houses
			},
		},
	},
	"state_properties": {
		"hour": True,
		"day": True,
		"solar_gain": True,
	},
	"power_grid_prop": {
		"base_power_mode" : "interpolation", # Interpolation (based on deadband bang-bang controller) or constant
		"base_power_parameters": {
			"constant" : {
				"avg_power_per_hvac": 4200,				# Per hvac. In Watts. 
				"init_signal_per_hvac": 910, 			# Per hvac.
			},
			"interpolation": {
				"path_datafile": "./monteCarlo/mergedGridSearchResultFinal.npy",
				"path_parameter_dict": "./monteCarlo/interp_parameters_dict.json",
				"path_dict_keys": "./monteCarlo/interp_dict_keys.csv",
				"interp_update_period": 300, 			# Seconds
				"interp_nb_agents": 100 					# Max number of agents over which the interpolation is run
			},		
		},
		"artificial_signal_ratio_range": 1, 			# Scale of artificial multiplicative factor randomly multiplied (or divided) at each episode during training. Ex: 1 will not modify signal. 3 will have signal between 33% and 300% of what is computed.
		"artificial_ratio": 1.0,

		"signal_mode": "regular_steps",					# Mode of the signal. Currently available: flat, sinusoidal, regular_steps
		"signal_parameters": {
			"flat": {

			},
			"sinusoidals": {
				"periods": [400, 1200],					# In seconds
				"amplitude_ratios": [0.1, 0.3],			# As a ratio of avg_power_per_hvac
			},
			"regular_steps": {
				"amplitude_per_hvac": 6000,					# In watts
				"period": 300,						# In seconds
			},
			"perlin": {
				"amplitude_ratios":0.9,
				"nb_octaves":5,
				"octaves_step":1,
				"period":400
			},
		}
	},
	"reward_prop": {
		"alpha_temp": 1,									# Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
		"alpha_sig": 1,									# Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
		"norm_reg_sig": 7500,							# Average power use, for signal normalization 
		"temp_penalty_mode": "individual_L2",				# Mode of temperature penalty
		"temp_penalty_parameters": {
			"individual_L2": {},
			"common_L2" : {},
			"common_max_error" : {},
			"mixture": {
				"alpha_ind_L2" : 1,
				"alpha_common_L2" : 1,
				"alpha_common_max" : 0,
			},
		},
		"sig_penalty_mode": "common_L2",					# Mode of signal penalty
	},
},

# Agent properties

"PPO_prop": {
    "actor_layers": [100, 100],
	"critic_layers": [100, 100],
    "gamma": 0.99,
    "lr_critic": 3e-3,
    "lr_actor": 1e-3,
    "clip_param": 0.2,
    "max_grad_norm": 0.5,
    "ppo_update_time": 10,
    "batch_size": 256,
    },

"MAPPO_prop": {
    "actor_layers": [100, 100],
	"critic_layers": [100, 100],
    "gamma": 0.99,
    "lr_critic": 3e-3,
    "lr_actor": 1e-3,
    "clip_param": 0.2,
    "max_grad_norm": 0.5,
    "ppo_update_time": 10,
    "batch_size": 256,
    },

"DQN_prop": {
    "network_layers": [100, 100],
    "gamma": 0.99,
    "tau": 0.01,
    "buffer_capacity": 524288,
    "lr": 1e-3,
    "batch_size": 256,
    "epsilon_decay": 0.99998,
    "min_epsilon": 0.01,
    },

"MPC_prop" : {
    "rolling_horizon": 15,
    },

}

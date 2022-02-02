
# House properties
# The house is modelled as a 10mx10m square with 2.5m height, 8 windows of 1.125 m² and 2 doors of 2m² each.
# The formulas for Ua, Cm, Ca and Hm are mainly taken here: http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide
default_house_prop = {
	"id": 1,
	"init_temp": 20,
	"target_temp": 20,
	"deadband": 2,
	"Ua" : 137.5*3,								# House walls conductance (W/K) (75 for walls and ceiling, 4.5 for two doors, 58 for windows). Multiplied by 3 to account for drafts (according to https://dothemath.ucsd.edu/2012/11/this-thermal-house/)
	"Cm" : 100*40700,							# House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)
	"Ca" : 3*100*2.5*1200,						# Air thermal mass in the house (J/K): 3 * (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)
	"Hm" : 455*8.14,							# House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.14 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))
}

noise_house_prop = {
	"std_target_temp": 1,     # Std Noise on target temperature
	"factor_thermo_low": 0.5,   # Lowest random factor for Ua, Cm, Ca, Hm
	"factor_thermo_high": 2,   # Highest random factor for Ua, Cm, Ca, Hm
}

# HVAC properties

default_hvac_prop = {
	"id": 1,
	"COP": 2.5,									# Coefficient of performance (power spent vs heat displaced)
	"cooling_capacity": 150,#1500,					# Cooling capacity (W) (by design, theoretical Ua * (max OD temp - target ID temp). Equivalent to 5200 BTU)
	"latent_cooling_fraction": 0.35,			# Fraction of latent cooling w.r.t. sensible cooling
	"lockout_duration": 4						# In seconds
}

noise_hvac_prop = {
	"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
	"factor_COP_low": 0.5,   # Lowest random factor for COP
	"factor_COP_high": 1.1,   # Highest random factor for COP
	"factor_cooling_capacity_low": 0.8,   # Lowest random factor for cooling_capacity
	"factor_cooling_capacity_high": 3,   # Highest random factor for cooling_capacity
}

# Env properties

default_env_properties = {
	"start_datetime": '2021-01-01 06:00:00',   	# Start date and time (Y-m-d H:M:S)
	"time_step": 4,							# Time step in seconds
	"cluster_properties": {
		"day_temp": 30,							# Day temperature
		"night_temp": 23,						# Night temperature
		"temp_std": 0.5,						# Noise std dev on the temperature
	},
	"power_grid_properties": {
		"avg_power_per_hvac": 29,					# Per hvac. In Watts. Based on average necessary power for bang-bang controller.
		"init_signal": 29, 							# Per hvac.
		"noise_mode": "none",					# Mode of the noise. Currently available: none, sinusoidal
		"noise_parameters": {
			"none": {},
			"sinusoidal": {
				"wavelength": 1200,					# In seconds
				"amplitude_per_hvac": 10,			# In watts per hvac
			},
		}
	},
	"alpha": 0, #1e-4,									# Balance parameter for loss function: temperature penalty + alpha * regulation signal penalty
}


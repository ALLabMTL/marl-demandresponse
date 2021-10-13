
# House properties

default_house_prop = {
	"id": 1,
	"init_temp": 20,
	"target_temp": 20,
	"deadband": 2,
	"Ua" : 84,									# House walls conductance (W/K) (default: 84W/K = 522 Btu/F/hr)
	"Cm" : 100*40700,							# House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)
	"Ca" : 100*2.5*1200,						# Air thermal mass in the house (J/K) (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)
	"Hm" : 455*8.29,							# House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.29 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))
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
	"cooling_capacity": 84*(30-20),				# Cooling capacity (W) (by design, Ua * (max OD temp - target ID temp))
	"latent_cooling_fraction": 0.35,			# Fraction of latent cooling w.r.t. sensible cooling
	"lockout_duration": 1						# In number of steps (TODO: change to seconds)
}

noise_hvac_prop = {
	"std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
	"factor_COP_low": 0.5,   # Lowest random factor for COP
	"factor_COP_high": 1.5,   # Highest random factor for COP
	"factor_cooling_capacity_low": 0.8,   # Lowest random factor for cooling_capacity
	"factor_cooling_capacity_high": 3,   # Highest random factor for cooling_capacity
}

# Env properties

default_env_properties = {
	"start_datetime": '2021-01-01 06:00:00',   	# Start date and time (Y-m-d H:M:S)
	"time_step": 3600,							# Time step in seconds
	"cluster_properties": {
		"day_temp": 30,							# Day temperature
		"night_temp": 23,						# Night temperature
		"temp_std": 0.5,						# Noise std dev on the temperature
	}

}


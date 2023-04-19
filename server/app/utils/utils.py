from typing import Literal, TypedDict

import numpy as np
from typing_extensions import NotRequired

from app.core.config.config import MarlConfig


def deadbandL2(target, deadband, value):
    if target + deadband / 2 < value:
        deadband_L2 = (value - (target + deadband / 2)) ** 2
    elif target - deadband / 2 > value:
        deadband_L2 = ((target - deadband / 2) - value) ** 2
    else:
        deadband_L2 = 0.0

    return deadband_L2


# TODO: change this method to something better to nromalize state dict
# maybe by making normalization in each class of the environment
# (hvac, building, cluster...)


class StateDict(TypedDict):
    indoor_temp: NotRequired[float]
    mass_temp: NotRequired[float]
    target_temp: NotRequired[float]
    deadband: NotRequired[float]
    OD_temp: NotRequired[float]
    Ua: NotRequired[float]
    Cm: NotRequired[float]
    Ca: NotRequired[float]
    Hm: NotRequired[float]
    cooling_capacity: NotRequired[float]
    cop: NotRequired[float]
    latent_cooling_fraction: NotRequired[float]
    datetime: NotRequired[float]
    solar_gain: NotRequired[float]
    seconds_since_off: NotRequired[float]
    lockout_duration: NotRequired[float]
    reg_signal: NotRequired[float]
    cluster_hvac_power: NotRequired[float]
    message: NotRequired[list]
    sin_day: NotRequired[float]
    cos_day: NotRequired[float]
    sin_hr: NotRequired[float]
    cos_hr: NotRequired[float]
    turned_on: NotRequired[Literal[0, 1]]
    lockout: NotRequired[Literal[0, 1]]


def normStateDict(sDict, config: MarlConfig, returnDict=False):
    default_house_prop = config.house_prop
    default_hvac_prop = config.hvac_prop
    default_env_prop = config.env_prop
    state_prop = config.env_prop.state_prop

    result: StateDict = {}

    k_temp = ["indoor_temp", "mass_temp", "target_temp"]
    k_div = ["cooling_capacity"]

    if state_prop.thermal:
        k_temp += ["OD_temp"]
        k_div += [
            "Ua",
            "Cm",
            "Ca",
            "Hm",
        ]

    if state_prop.hvac:
        k_div += [
            "cop",
            "latent_cooling_fraction",
        ]

    # k_lockdown = ['hvac_seconds_since_off', 'hvac_lockout_duration']
    for k in k_temp:
        # Assuming the temperatures will be between 15 to 30, centered around 20 -> between -1 and 2, centered around 0.
        result[k] = (sDict[k] - 20) / 5
    result["deadband"] = sDict["deadband"]

    if state_prop.day:
        day = sDict["datetime"].timetuple().tm_yday
        result["sin_day"] = np.sin(day * 2 * np.pi / 365)
        result["cos_day"] = np.cos(day * 2 * np.pi / 365)
    if state_prop.hour:
        hour = sDict["datetime"].hour
        result["sin_hr"] = np.sin(hour * 2 * np.pi / 24)
        result["cos_hr"] = np.cos(hour * 2 * np.pi / 24)

    if state_prop.solar_gain:
        result["solar_gain"] = sDict["solar_gain"] / 1000

    for k in k_div:
        if k in list(default_house_prop.dict().keys()):
            result[k] = sDict[k] / default_house_prop.dict()[k]
        elif k in list(default_hvac_prop.dict().keys()):
            result[k] = sDict[k] / default_hvac_prop.dict()[k]
        else:
            raise Exception(f"Error Key Matching. {k}")

    result["turned_on"] = 1 if sDict["turned_on"] else 0
    result["lockout"] = 1 if sDict["lockout"] else 0

    result["seconds_since_off"] = sDict["seconds_since_off"] / sDict["lockout_duration"]
    result["lockout_duration"] = sDict["lockout_duration"] / sDict["lockout_duration"]

    result["reg_signal"] = sDict["reg_signal"] / (
        default_env_prop.reward_prop.norm_reg_sig
        * default_env_prop.cluster_prop.nb_agents
    )
    result["cluster_hvac_power"] = sDict["cluster_hvac_power"] / (
        default_env_prop.reward_prop.norm_reg_sig
        * default_env_prop.cluster_prop.nb_agents
    )

    temp_messages = []
    for message in sDict["message"]:
        r_message = {}
        r_message["current_temp_diff_to_target"] = (
            message["current_temp_diff_to_target"] / 5
        )  # Already a difference, only need to normalize like k_temps
        r_message["hvac_seconds_since_off"] = (
            message["hvac_seconds_since_off"] / sDict["lockout_duration"]
        )
        r_message["hvac_curr_consumption"] = (
            message["hvac_curr_consumption"] / default_env_prop.reward_prop.norm_reg_sig
        )
        r_message["hvac_max_consumption"] = (
            message["hvac_max_consumption"] / default_env_prop.reward_prop.norm_reg_sig
        )

        if default_env_prop.message_properties.thermal:
            r_message["Ua"] = message["house_Ua"] / default_house_prop.Ua
            r_message["Cm"] = message["house_Cm"] / default_house_prop.Cm
            r_message["Ca"] = message["house_Ca"] / default_house_prop.Ca
            r_message["Hm"] = message["house_Hm"] / default_house_prop.Hm
        if default_env_prop.message_properties.hvac:
            r_message["cop"] = message["cop"] / default_hvac_prop.cop
            r_message["latent_cooling_fraction"] = (
                message["latent_cooling_fraction"]
                / default_hvac_prop.latent_cooling_fraction
            )
            r_message["cooling_capacity"] = (
                message["cooling_capacity"] / default_hvac_prop.cooling_capacity
            )
        temp_messages.append(r_message)
    if returnDict:
        result["message"] = temp_messages

    else:  # Flatten the dictionary in a single np_array
        flat_messages = []
        for message in temp_messages:
            flat_message = list(message.values())
            flat_messages = flat_messages + flat_message
        result = np.array(list(result.values()) + flat_messages)

    return result

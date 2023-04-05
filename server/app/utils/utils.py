from datetime import datetime

import numpy as np


def deadbandL2(target, deadband, value):
    if target + deadband / 2 < value:
        deadband_L2 = (value - (target + deadband / 2)) ** 2
    elif target - deadband / 2 > value:
        deadband_L2 = ((target - deadband / 2) - value) ** 2
    else:
        deadband_L2 = 0.0

    return deadband_L2


def sort_dict_keys(point, dict_keys):
    sorted_point = {}
    for key in dict_keys:
        sorted_point[key] = point[key]
    return sorted_point


def compute_solar_gain(
    date_time: datetime, window_area: float, shading_coeff: float
) -> float:
    """
    Computes the solar gain, i.e. the heat transfer received from the sun through the windows.

    Return:
    solar_gain: float, direct solar radiation passing through the windows at a given moment in Watts

    Parameters
    date_time: datetime, current date and time

    ---
    Source and assumptions:
    CIBSE. (2015). Environmental Design - CIBSE Guide A (8th Edition) - 5.9.7 Solar Cooling Load Tables. CIBSE.
    Retrieved from https://app.knovel.com/hotlink/pdf/id:kt0114THK9/environmental-design/solar-cooling-load-tables
    Table available: https://www.cibse.org/Knowledge/Guide-A-2015-Supplementary-Files/Chapter-5

    Coefficient obtained by performing a polynomial regression on the table "solar cooling load at stated sun time at latitude 30".

    Based on the following assumptions.
    - Latitude is 30. (The latitude of Austin in Texas is 30.266666)
    - The SCL before 7:30 and after 17:30 is negligible for latitude 30.
    - The windows are distributed perfectly evenly around the building.
    - There are no horizontal windows, for example on the roof.
    """

    x = date_time.hour + date_time.minute / 60 - 7.5
    if x < 0 or x > 10:
        solar_cooling_load = 0
    else:
        y = date_time.month + date_time.day / 30 - 1
        coeff = [
            4.36579418e01,
            1.58055357e02,
            8.76635241e01,
            -4.55944821e01,
            3.24275366e00,
            -4.56096472e-01,
            -1.47795612e01,
            4.68950855e00,
            -3.73313090e01,
            5.78827663e00,
            1.04354810e00,
            2.12969604e-02,
            2.58881400e-03,
            -5.11397219e-04,
            1.56398008e-02,
            -1.18302764e-01,
            -2.71446436e-01,
            -3.97855577e-02,
        ]

        solar_cooling_load = (
            coeff[0]
            + x * coeff[1]
            + y * coeff[2]
            + x**2 * coeff[3]
            + x**2 * y * coeff[4]
            + x**2 * y**2 * coeff[5]
            + y**2 * coeff[6]
            + x * y**2 * coeff[7]
            + x * y * coeff[8]
            + x**3 * coeff[9]
            + y**3 * coeff[10]
            + x**3 * y * coeff[11]
            + x**3 * y**2 * coeff[12]
            + x**3 * y**3 * coeff[13]
            + x**2 * y**3 * coeff[14]
            + x * y**3 * coeff[15]
            + x**4 * coeff[16]
            + y**4 * coeff[17]
        )

    solar_gain = window_area * shading_coeff * solar_cooling_load
    return solar_gain

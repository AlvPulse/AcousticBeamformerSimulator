import numpy as np
try:
    from acoustics.atmosphere import Atmosphere
    HAS_ACOUSTICS = True
except ImportError:
    HAS_ACOUSTICS = False

def iso9613_absorption_db_per_m(freq_hz, temp_c=20.0, rel_humidity=50.0, pressure_kpa=101.325) -> float:
    """
    Computes ISO 9613-1 pure-tone atmospheric absorption coefficient (dB/m).
    """
    if HAS_ACOUSTICS:
        # python-acoustics implements ISO 9613-1 correctly.
        atm = Atmosphere(temperature=temp_c, relative_humidity=rel_humidity,
                         pressure=pressure_kpa * 1000.0) # pressure in Pa for acoustics

        # acoustics returns attenuation in dB/m.
        # attenuation_coefficient returns dB/m.
        # Make sure freq is numpy array or list if required, but it handles scalars
        alpha = atm.attenuation_coefficient(np.array([freq_hz]))[0]
        return float(alpha)

    # Manual fallback if acoustics is not installed
    T = temp_c + 273.15 # K
    Tr = 293.15 # K
    T01 = 273.16 # K
    pr = 101.325 # kPa
    pa = pressure_kpa # kPa
    h_rel = rel_humidity # %
    f = freq_hz

    psat_pr = 10**(-6.8346 * (T01/T)**1.261 + 4.6151)
    h = h_rel * (psat_pr) / (pa/pr)

    frO = (pa/pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa/pr) * (T/Tr)**-0.5 * (9 + 280 * h * np.exp(-4.170 * ((T/Tr)**(-1/3) - 1)))

    term1 = 0.01275 * np.exp(-2239.1/T) * (frO + f**2 / frO)**-1
    term2 = 0.1068 * np.exp(-3352.0/T) * (frN + f**2 / frN)**-1

    alpha = 8.686 * f**2 * ( 1.84e-11 * (pr/pa) * (T/Tr)**0.5 + (T/Tr)**-2.5 * (term1 + term2) )
    return float(alpha)

def path_loss_db(freq_hz, distance_m, temp_c=20.0, rel_humidity=50.0, pressure_kpa=101.325,
                 ground_effect_loss_db=0.0, shadowing_std_db=0.0) -> float:
    """
    Computes total path loss in dB = geometric spreading + atmospheric absorption
    + ground effect + log-normal shadowing.
    PL(f, r) = 20*log10(r / r0) + alpha(f)*r + A_ground + X_sigma
    """
    if distance_m <= 0:
        return 0.0 # No path loss at 0m, or define arbitrarily

    r0 = 1.0
    spreading_loss = 20.0 * np.log10(distance_m / r0)

    alpha = iso9613_absorption_db_per_m(freq_hz, temp_c, rel_humidity, pressure_kpa)
    abs_loss = alpha * distance_m

    # Shadowing (log-normal random variable added in dB domain)
    shadowing_loss = 0.0
    if shadowing_std_db > 0.0:
        shadowing_loss = np.random.normal(0, shadowing_std_db)

    total_loss = spreading_loss + abs_loss + ground_effect_loss_db + shadowing_loss
    return total_loss

def cartesian_to_spherical(x, y, z):
    """Returns r, az, el (az, el in radians)"""
    r = np.sqrt(x**2 + y**2 + z**2)
    # az = atan2(y, x), CCW positive from +x
    az = np.arctan2(y, x)
    # el = asin(z/r), positive above xy plane
    el = np.arcsin(z / r) if r > 0 else 0.0
    return r, az, el

def spherical_to_cartesian(r, az_deg, el_deg):
    """
    az_deg: azimuth in degrees, [-180, 180), +x axis, CCW positive
    el_deg: elevation in degrees, [-90, 90], positive above xy plane
    """
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)

    return np.array([x, y, z])

def per_sensor_delays(array, az_deg, el_deg, r=None, c=343.0):
    """
    Computes exact per-sensor delay: tau_i = |p_source - p_i| / c
    If r is None, calculates far-field plane wave delay relative to origin.
    Returns:
        delays: np.ndarray shape (n_sensors,)
    """
    pos = array.positions # (n_sensors, 3)

    if r is None or np.isinf(r):
        # Far-field plane wave delay: tau_i = - (p_i dot k_hat) / c
        # k_hat is unit vector pointing towards the source
        k_hat = spherical_to_cartesian(1.0, az_deg, el_deg)
        delays = -np.dot(pos, k_hat) / c
        # We shift them so min delay is 0 or just leave as relative.
        # Typically relative delays around 0 are fine.
    else:
        # Near-field exact delay
        p_source = spherical_to_cartesian(r, az_deg, el_deg)
        diffs = p_source - pos # vectors from sensor to source? Or source to sensor?
        # Distance from source to sensor
        distances = np.linalg.norm(diffs, axis=1)
        delays = distances / c

    return delays

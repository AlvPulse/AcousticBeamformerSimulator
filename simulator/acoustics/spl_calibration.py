import numpy as np

def calibrate_signal_spl(signal, target_spl_db):
    """
    Scales a digital signal to represent a specific Sound Pressure Level (SPL) in dB.
    Assumes 0 dB SPL = 2e-5 Pascals RMS.
    We normalize the digital signal such that its RMS matches the physical pressure.
    """
    p_ref = 2e-5 # Reference pressure in Pascals
    target_pressure_rms = p_ref * (10 ** (target_spl_db / 20.0))

    current_rms = np.sqrt(np.mean(signal**2))
    if current_rms == 0:
        return signal

    gain = target_pressure_rms / current_rms
    return signal * gain

def calculate_received_spl(signal):
    """
    Calculates the SPL in dB of a given pressure signal array.
    """
    p_ref = 2e-5
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms / p_ref)

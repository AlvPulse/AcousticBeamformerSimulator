import numpy as np

def compute_si_sdr(reference, estimate):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    """
    # Ensure vectors
    reference = np.array(reference).flatten()
    estimate = np.array(estimate).flatten()

    # Target power
    ref_energy = np.sum(reference**2)
    if ref_energy == 0:
        return 0.0

    # Optimal scaling factor
    alpha = np.sum(reference * estimate) / ref_energy
    target = alpha * reference

    # Error
    e_res = estimate - target

    target_power = np.sum(target**2)
    e_res_power = np.sum(e_res**2)

    if e_res_power == 0:
        return 100.0 # Perfect

    si_sdr = 10 * np.log10(target_power / e_res_power)
    return si_sdr

def compute_lsd(clean_stft, beam_stft):
    """
    Log-Spectral Distance.
    Inputs are STFT magnitudes: (F, T)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    power_clean = np.abs(clean_stft)**2 + eps
    power_beam = np.abs(beam_stft)**2 + eps

    # 10 log10 (P_clean / P_beam)
    log_ratio = 10 * np.log10(power_clean / power_beam)

    # Mean over frequency
    mean_sq_freq = np.mean(log_ratio**2, axis=0)

    # Sqrt and mean over time
    lsd = np.mean(np.sqrt(mean_sq_freq))
    return lsd

def compute_array_gain(snr_in_linear, snr_out_linear):
    """
    Array Gain (dB)
    """
    return 10 * np.log10(snr_out_linear / (snr_in_linear + 1e-12))

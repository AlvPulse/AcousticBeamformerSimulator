import numpy as np

def compute_si_sdr(reference, estimate):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    """
    # Ensure zero mean
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)

    # Scale invariant
    alpha = np.sum(ref * est) / (np.sum(ref**2) + 1e-8)
    e_target = alpha * ref
    e_res = est - e_target

    # SNR
    num = np.sum(e_target**2)
    den = np.sum(e_res**2) + 1e-8

    return 10 * np.log10(num / den)

def compute_lsd(reference, estimate, n_fft=512, hop_length=256):
    """
    Log-Spectral Distance (LSD)
    """
    import scipy.signal

    _, _, Z_ref = scipy.signal.stft(reference, nperseg=n_fft, noverlap=n_fft-hop_length)
    _, _, Z_est = scipy.signal.stft(estimate, nperseg=n_fft, noverlap=n_fft-hop_length)

    S_ref = np.abs(Z_ref)**2
    S_est = np.abs(Z_est)**2

    S_ref = np.maximum(S_ref, 1e-10)
    S_est = np.maximum(S_est, 1e-10)

    log_ratio = 10 * np.log10(S_ref / S_est)

    # Average over frequency (k) first, then time (t)
    lsd_per_frame = np.sqrt(np.mean(log_ratio**2, axis=0))
    lsd_total = np.mean(lsd_per_frame)

    return lsd_total

def compute_sir(target_signal, interferer_signal, beamformer, array_positions, target_az, target_el):
    """
    Computes Signal-to-Interference Ratio improvement.
    Note: Requires processing target and interferer separately through the fixed beamformer weights.
    (Simplified here: standard SIR compares target power to interferer power at output).
    """
    p_sig = np.mean(target_signal**2)
    p_int = np.mean(interferer_signal**2)
    if p_int == 0: return 100.0
    return 10 * np.log10(p_sig / (p_int + 1e-8))

def compute_coherent_gain(raw_center_mic, beamformed_out, target_multi, noise_multi):
    """
    Computes Input SNR, Output SNR, and True Spatial Array Gain (AG = SNR_out - SNR_in)
    to prove coherent combining vs non-coherent noise accumulation.
    """
    p_target_in = np.mean(target_multi[0]**2)
    p_noise_in = np.mean(noise_multi[0]**2)

    snr_in = 10 * np.log10(p_target_in / (p_noise_in + 1e-10))

    # Scale invariant
    ref = target_multi[0] - np.mean(target_multi[0])
    est = beamformed_out - np.mean(beamformed_out)

    alpha = np.sum(ref * est) / (np.sum(ref**2) + 1e-8)
    e_target = alpha * ref
    e_res = est - e_target

    p_target_out = np.mean(e_target**2)
    p_noise_out = np.mean(e_res**2)

    snr_out = 10 * np.log10(p_target_out / (p_noise_out + 1e-10))
    array_gain = snr_out - snr_in

    return {
        "snr_in_db": snr_in,
        "snr_out_db": snr_out,
        "array_gain_db": array_gain
    }

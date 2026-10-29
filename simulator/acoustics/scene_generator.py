import numpy as np
import scipy.signal as signal
from simulator.acoustics.channel import compute_delay_and_attenuation, generate_wind_diffuse_noise

def compute_rms(sig):
    return np.sqrt(np.mean(sig**2))

def synthesize_scene(pos_mics, fs, n_samples, target_sig, target_pos, interferer_sig, interferer_pos_trajectory,
                     target_snr_db, interferer_inr_db, turbulence=False):
    """
    Generates a multi-channel mixture.
    target_pos: (3,)
    interferer_pos_trajectory: (N, 3) or (3,)
    """
    M = pos_mics.shape[0]

    # 1. Background Noise (White + Diffuse Wind)
    white_noise = np.random.randn(M, n_samples)
    diffuse_noise = generate_wind_diffuse_noise(pos_mics, n_samples, fs)
    # Combine noises (equal power)
    white_noise /= compute_rms(white_noise)
    diffuse_noise /= compute_rms(diffuse_noise)
    background_noise = (white_noise + diffuse_noise) / np.sqrt(2)
    noise_power = compute_rms(background_noise[0])**2

    # 2. Target Signal
    target_delays, target_atts = compute_delay_and_attenuation(pos_mics, target_pos, is_spherical=False)
    target_mc = np.zeros((M, n_samples))
    for m in range(M):
        # Shift
        shift_samples = int(np.round(target_delays[m] * fs))
        shifted = np.roll(target_sig, shift_samples)
        if shift_samples > 0:
            shifted[:shift_samples] = 0
        elif shift_samples < 0:
            shifted[shift_samples:] = 0
        target_mc[m] = shifted * target_atts[m]

    # Scale target to achieve target_snr_db at mic 0
    target_power = compute_rms(target_mc[0])**2
    target_scale = np.sqrt(noise_power * (10**(target_snr_db/10)) / (target_power + 1e-12))
    target_mc *= target_scale

    # 3. Interferer Signal
    interferer_mc = np.zeros((M, n_samples))

    if interferer_pos_trajectory.ndim == 1:
        # Static
        int_delays, int_atts = compute_delay_and_attenuation(pos_mics, interferer_pos_trajectory, is_spherical=True)
        for m in range(M):
            shift_samples = int(np.round(int_delays[m] * fs))
            shifted = np.roll(interferer_sig, shift_samples)
            if shift_samples > 0:
                shifted[:shift_samples] = 0
            elif shift_samples < 0:
                shifted[shift_samples:] = 0
            interferer_mc[m] = shifted * int_atts[m]
    else:
        # Dynamic (trajectory) - simple approximation by updating delay per block or sample
        # For simplicity in this benchmark, we can do it block-wise or use a fractional delay filter.
        # Let's use a simpler static interferer at average position for now, or block-wise if needed.
        # Given the instruction to handle trajectory, let's implement block-wise processing.
        block_size = 256
        n_blocks = n_samples // block_size
        for b in range(n_blocks):
            idx = b * block_size
            pos_idx = min(idx, interferer_pos_trajectory.shape[0]-1)
            pos_current = interferer_pos_trajectory[pos_idx]

            int_delays, int_atts = compute_delay_and_attenuation(pos_mics, pos_current, is_spherical=True)
            for m in range(M):
                shift_samples = int(np.round(int_delays[m] * fs))

                # We need the source chunk that will arrive at time idx:idx+block_size
                # Because of delay, we need source samples at idx - shift_samples
                src_start = idx - shift_samples
                src_end = src_start + block_size

                if src_start >= 0 and src_end < n_samples:
                    chunk = interferer_sig[src_start:src_end]
                else:
                    chunk = np.zeros(block_size)
                    if src_start < 0:
                        valid_start = 0
                        valid_end = min(src_end, n_samples)
                        if valid_end > 0:
                            chunk[-src_start: -src_start + valid_end] = interferer_sig[valid_start:valid_end]
                    else:
                        valid_start = src_start
                        valid_end = n_samples
                        if valid_start < n_samples:
                            chunk[:valid_end-valid_start] = interferer_sig[valid_start:valid_end]

                interferer_mc[m, idx:idx+block_size] = chunk * int_atts[m]

    # Scale interferer
    int_power = compute_rms(interferer_mc[0])**2
    int_scale = np.sqrt(noise_power * (10**(interferer_inr_db/10)) / (int_power + 1e-12))
    interferer_mc *= int_scale

    mix = target_mc + interferer_mc + background_noise
    return mix, target_mc, interferer_mc, background_noise

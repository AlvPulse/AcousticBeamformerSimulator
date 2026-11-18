import numpy as np
import scipy.signal as signal
from simulator.acoustics.channel import compute_delay_and_attenuation, generate_wind_diffuse_noise

def compute_rms(sig):
    return np.sqrt(np.mean(sig**2))


def fractional_delay(sig, delay_s, fs):
    """
    Exact band-limited delay via FFT phase shift: returns s(t - delay_s).
    Replaces integer-sample np.roll, which quantized sub-sample delays and
    injected up to ~0.8 rad of phase error per mic at 2 kHz.
    """
    n = len(sig)
    S = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(S * np.exp(-2j * np.pi * freqs * delay_s), n=n)

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
        target_mc[m] = fractional_delay(target_sig, target_delays[m], fs) * target_atts[m]

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
            interferer_mc[m] = fractional_delay(interferer_sig, int_delays[m], fs) * int_atts[m]
    else:
        # Dynamic (trajectory): time-varying fractional delay via linear
        # interpolation of the source at sample positions (t_out - tau(t)).
        block_size = 256
        n_blocks = n_samples // block_size
        src_index = np.arange(n_samples)
        for b in range(n_blocks):
            idx = b * block_size
            pos_idx = min(idx, interferer_pos_trajectory.shape[0]-1)
            pos_current = interferer_pos_trajectory[pos_idx]

            int_delays, int_atts = compute_delay_and_attenuation(pos_mics, pos_current, is_spherical=True)
            out_t = np.arange(idx, idx + block_size) / fs
            for m in range(M):
                src_pos = (out_t - int_delays[m]) * fs
                valid = (src_pos >= 0) & (src_pos < n_samples - 1)
                chunk = np.zeros(block_size)
                if np.any(valid):
                    chunk[valid] = np.interp(src_pos[valid], src_index, interferer_sig)

                interferer_mc[m, idx:idx+block_size] = chunk * int_atts[m]

    # Scale interferer
    int_power = compute_rms(interferer_mc[0])**2
    int_scale = np.sqrt(noise_power * (10**(interferer_inr_db/10)) / (int_power + 1e-12))
    interferer_mc *= int_scale

    mix = target_mc + interferer_mc + background_noise
    return mix, target_mc, interferer_mc, background_noise

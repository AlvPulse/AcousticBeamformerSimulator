import numpy as np
import librosa
import scipy.signal
from simulator.acoustics.channel import apply_greens_function, generate_diffuse_noise, apply_turbulence_decoherence
from simulator.acoustics.array import generate_log_spiral_array

def calculate_snr_gain(signal, noise, target_snr_db):
    """
    Calculates the scalar gain needed to achieve target_snr_db.
    """
    p_sig = np.mean(signal**2)
    p_noise = np.mean(noise**2)
    if p_sig == 0 or p_noise == 0:
        return 1.0

    snr_linear = 10**(target_snr_db / 10.0)
    gain = np.sqrt((snr_linear * p_noise) / p_sig)
    return gain

def generate_scene(target_audio_path, array_positions, fs=16000,
                   target_az=0.0, target_el=0.0, target_dist=120.0, target_snr=-15.0,
                   interferer_inr=35.0, interferer_speed=15.0,
                   turbulence_sigma=0.0, duration_sec=3.0,
                   num_random_interferers=1):
    """
    Mixes a distant target, near-field dynamic interferer, random interferers, and noise.
    """
    n_sensors = array_positions.shape[0]
    n_samples = int(fs * duration_sec)

    # 1. Load Distant Target Signal
    s_t, _ = librosa.load(target_audio_path, sr=fs)
    if len(s_t) > n_samples:
        s_t = s_t[:n_samples]
    else:
        s_t = np.pad(s_t, (0, n_samples - len(s_t)))

    # Calculate target Cartesian position
    az_rad = np.deg2rad(target_az)
    el_rad = np.deg2rad(target_el)
    target_pos = target_dist * np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])

    # Target Propagation
    target_multi = apply_greens_function(s_t, fs, target_pos, array_positions)

    # Apply Atmospheric Turbulence
    if turbulence_sigma > 0:
        target_multi = apply_turbulence_decoherence(target_multi, fs, array_positions, turbulence_sigma, target_dist)

    # 2. Background Noise Generation (Uncorrelated + Diffuse)
    # Uncorrelated Thermal Noise
    thermal_noise = np.random.randn(n_sensors, n_samples)

    # Spatially Diffuse Wind Noise
    diffuse_noise = generate_diffuse_noise(array_positions, fs, n_samples)

    # Combine noise (50% thermal, 50% diffuse power)
    background_noise = thermal_noise + diffuse_noise
    background_noise /= np.std(background_noise) # Unit variance

    # 3. Scale Target to desired SNR
    # Measure SNR at the center mic (assume mic 0 is roughly center or use mean power)
    if target_snr == "dynamic":
        # Dynamic SNR sweep: Airplane approaching, SNR sweeps from -25 dB to +5 dB over the duration
        snr_start = -25.0
        snr_end = 5.0

        # Calculate instantaneous gains
        # Using a smooth linear ramp in dB space
        db_sweep = np.linspace(snr_start, snr_end, n_samples)

        # Calculate raw powers
        p_noise = np.mean(background_noise[0]**2)
        p_sig_raw = np.mean(target_multi[0]**2)

        if p_sig_raw > 0 and p_noise > 0:
            # Need to apply time-varying gain
            gain_curve = np.sqrt((10**(db_sweep / 10.0) * p_noise) / p_sig_raw)
            # Apply to all mics
            target_multi *= gain_curve[np.newaxis, :]
    else:
        target_gain = calculate_snr_gain(target_multi[0], background_noise[0], target_snr)
        target_multi *= target_gain

    # 4. Main Near-Field Dynamic Interferer (e.g. passing car)
    i_t = np.random.randn(n_samples) # Broadband vehicle passing sound approximation
    # Apply a lowpass filter to mimic car rumble
    b, a = scipy.signal.butter(4, 1000/(fs/2), btype='low')
    i_t = scipy.signal.lfilter(b, a, i_t)

    # Dynamic trajectory: Passing at distance y=8m, moving along x-axis
    # speed v in m/s
    x_traj = np.linspace(-interferer_speed * duration_sec / 2, interferer_speed * duration_sec / 2, n_samples)
    interferer_multi = np.zeros((n_sensors, n_samples))

    # Frame-based dynamic application (approximation since exact trajectory Green's is time-varying)
    hop_size = 512
    for hop_idx in range(0, n_samples, hop_size):
        end_idx = min(hop_idx + hop_size, n_samples)
        if hop_idx >= end_idx: break

        mid_idx = (hop_idx + end_idx) // 2
        car_pos = np.array([x_traj[mid_idx], 8.0, 0.0]) # Pass by at 8 meters

        chunk = i_t[hop_idx:end_idx]
        chunk_multi = apply_greens_function(chunk, fs, car_pos, array_positions)
        interferer_multi[:, hop_idx:end_idx] = chunk_multi

    # Scale Main Interferer to INR
    interf_gain = calculate_snr_gain(interferer_multi[0], background_noise[0], interferer_inr)
    interferer_multi *= interf_gain

    # 5. Multiple Random Interferers (For S7)
    random_interferers_multi = np.zeros((n_sensors, n_samples))
    for r in range(num_random_interferers - 1): # -1 because the main dynamic car counts as 1
        r_sig = np.random.randn(n_samples)

        r_az = np.random.uniform(-180, 180)
        r_el = np.random.uniform(0, 30)
        r_dist = np.random.uniform(20, 80)

        r_pos = r_dist * np.array([
            np.cos(np.deg2rad(r_el)) * np.cos(np.deg2rad(r_az)),
            np.cos(np.deg2rad(r_el)) * np.sin(np.deg2rad(r_az)),
            np.sin(np.deg2rad(r_el))
        ])

        r_multi = apply_greens_function(r_sig, fs, r_pos, array_positions)
        # Random INR between 10 and 25 dB
        r_gain = calculate_snr_gain(r_multi[0], background_noise[0], np.random.uniform(10, 25))
        r_multi *= r_gain
        random_interferers_multi += r_multi

    # Final Mixture
    mixture = target_multi + interferer_multi + random_interferers_multi + background_noise

    return mixture, target_multi, interferer_multi + random_interferers_multi, background_noise

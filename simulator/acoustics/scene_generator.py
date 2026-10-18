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
                   num_random_interferers=1, cocktail_party=False, wind_gusts=False,
                   sensor_failure=False, moving_target=False):
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

    if moving_target:
        # Target moves horizontally across the field of view (e.g. from -15 deg to +15 deg)
        # over the duration of the clip.
        az_traj = np.linspace(target_az - 15.0, target_az + 15.0, n_samples)
        el_rad = np.deg2rad(target_el)
        target_multi = np.zeros((n_sensors, n_samples))

        t_idx = np.arange(n_samples)
        for m in range(n_sensors):
            # Calculate time-varying Cartesian position
            az_rad = np.deg2rad(az_traj)
            x = target_dist * np.cos(el_rad) * np.cos(az_rad)
            y = target_dist * np.cos(el_rad) * np.sin(az_rad)
            z = target_dist * np.sin(el_rad) * np.ones_like(az_rad)

            dx = x - array_positions[m, 0]
            dy = y - array_positions[m, 1]
            dz = z - array_positions[m, 2]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            atten = 1.0 / (4 * np.pi * dist)
            delay_samples = (dist / 343.0) * fs
            lookup_idx = t_idx - delay_samples
            valid = lookup_idx >= 0

            shifted_sig = np.zeros(n_samples)
            shifted_sig[valid] = np.interp(lookup_idx[valid], t_idx, s_t)
            target_multi[m, :] = shifted_sig * atten
    else:
        # Calculate static target Cartesian position
        az_rad = np.deg2rad(target_az)
        el_rad = np.deg2rad(target_el)
        target_pos = target_dist * np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        # Target Propagation
        target_multi_padded = apply_greens_function(s_t, fs, target_pos, array_positions)
        # Target is static, so we can just crop the valid window from the start
        target_multi = target_multi_padded[:, :n_samples]

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

    # Add severe Wind Gusts if requested
    # Wind gusts are loud, transient, low-frequency pressure spikes that hit microphones independently
    if wind_gusts:
        for m in range(n_sensors):
            # Create a 1/f noise burst
            gust_noise = np.random.randn(n_samples)
            b, a = scipy.signal.butter(2, 50/(fs/2), btype='low') # Very low frequency rumble
            gust_noise = scipy.signal.lfilter(b, a, gust_noise)

            # Apply a harsh envelope (transient spike)
            env = np.zeros(n_samples)
            # 2 random gusts per microphone
            for _ in range(2):
                start = np.random.randint(0, int(n_samples * 0.8))
                length = np.random.randint(int(fs * 0.2), int(fs * 0.8))

                # Make sure we don't go out of bounds
                actual_length = min(length, n_samples - start)
                if actual_length <= 0: continue

                window = scipy.signal.windows.hann(actual_length)
                env[start:start+actual_length] += window

            gust_noise *= env
            # Gusts are extremely loud (+40 dB relative to background noise)
            gust_gain = calculate_snr_gain(gust_noise, background_noise[0], 40.0)
            background_noise[m] += gust_noise * gust_gain

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

    # Continuous time-varying fractional delay using linear interpolation
    # This prevents the 512-sample block truncation artifacts.
    t_idx = np.arange(n_samples)

    for m in range(n_sensors):
        # Calculate time-varying distance from mic m to the moving car
        dx = x_traj - array_positions[m, 0]
        dy = 8.0 - array_positions[m, 1]
        dz = 0.0 - array_positions[m, 2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Time-varying attenuation
        atten = 1.0 / (4 * np.pi * dist)

        # Time-varying delay in samples
        delay_samples = (dist / 343.0) * fs

        # We need to look up the source signal at (t - delay)
        lookup_idx = t_idx - delay_samples

        # Linear interpolation for fractional delay
        # Mask out indices that fall before 0
        valid = lookup_idx >= 0

        # Interpolate
        # np.interp expects increasing x-coordinates, which lookup_idx is.
        # But we are mapping the current time t to the source value at lookup_idx.
        shifted_sig = np.zeros(n_samples)
        shifted_sig[valid] = np.interp(lookup_idx[valid], t_idx, i_t)

        interferer_multi[m, :] = shifted_sig * atten

    # Scale Main Interferer to INR
    interf_gain = calculate_snr_gain(interferer_multi[0], background_noise[0], interferer_inr)
    interferer_multi *= interf_gain

    # 5. Multiple Random Interferers (For S7 / S9 Cocktail Party)
    random_interferers_multi = np.zeros((n_sensors, n_samples))
    for r in range(num_random_interferers - 1): # -1 because the main dynamic car counts as 1
        if cocktail_party:
            # Shift target signal by random amount to create a "babble/cocktail" effect
            shift = np.random.randint(int(fs * 0.2), int(fs * 1.5))
            r_sig = np.roll(s_t, shift)
        else:
            r_sig = np.random.randn(n_samples)

        r_az = np.random.uniform(-180, 180)
        r_el = np.random.uniform(0, 30)
        r_dist = np.random.uniform(20, 80)

        r_pos = r_dist * np.array([
            np.cos(np.deg2rad(r_el)) * np.cos(np.deg2rad(r_az)),
            np.cos(np.deg2rad(r_el)) * np.sin(np.deg2rad(r_az)),
            np.sin(np.deg2rad(r_el))
        ])

        r_multi_padded = apply_greens_function(r_sig, fs, r_pos, array_positions)
        # Static, just take the first n_samples
        r_multi = r_multi_padded[:, :n_samples]

        # Random INR between 10 and 25 dB
        r_gain = calculate_snr_gain(r_multi[0], background_noise[0], np.random.uniform(10, 25))
        r_multi *= r_gain
        random_interferers_multi += r_multi

    # Final Mixture
    mixture = target_multi + interferer_multi + random_interferers_multi + background_noise

    if sensor_failure:
        # Simulate hardware failure where 10% of the microphones randomly die (output zero)
        # or physically clip due to buffeting.
        fail_count = max(1, int(n_sensors * 0.1))
        fail_idx = np.random.choice(n_sensors, fail_count, replace=False)
        for idx in fail_idx:
            # 50% chance to die, 50% chance to clip to max rail
            if np.random.rand() > 0.5:
                mixture[idx, :] = 0.0
            else:
                max_val = np.max(np.abs(mixture))
                mixture[idx, :] = np.clip(mixture[idx, :] * 100, -max_val, max_val)

    return mixture, target_multi, interferer_multi + random_interferers_multi, background_noise

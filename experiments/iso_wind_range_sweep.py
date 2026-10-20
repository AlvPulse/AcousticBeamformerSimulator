import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import sys

sys.path.append('.')
from simulator.acoustics.array import generate_log_spiral_array
from simulator.acoustics.channel import apply_greens_function
from simulator.beamformers import TaperedDAS, RobustLCMV, GSC_MCWF
from simulator.evaluation import evaluate_lam_preservation

def generate_drone_signature(fs, duration_sec):
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    f0 = 400.0
    sig = np.zeros_like(t)
    for i in range(1, 6):
        sig += np.sin(2 * np.pi * f0 * i * t) * (1.0 / i)
    return sig

def generate_airplane_signature(fs, duration_sec):
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 200/(fs/2), btype='low')
    rumble = signal.filtfilt(b, a, noise)
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 20 * t))
    return rumble * modulation

def generate_car_signature(fs, duration_sec):
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    noise = np.random.randn(len(t))
    b, a = signal.butter(4, [100/(fs/2), 1500/(fs/2)], btype='band')
    return signal.filtfilt(b, a, noise)

def run_2d_sweep():
    print("==================================================")
    print("Executing 2D ISO 9613-1 Range vs Wind Sweep")
    print("==================================================")

    fs = 16000
    duration_sec = 1.0
    N = int(fs * duration_sec)
    pos = generate_log_spiral_array(n_sensors=50, aperture_m=2.5)

    targets = {
        "Drone (High Freq Harmonics)": generate_drone_signature(fs, duration_sec),
        "Airplane (Low Freq Rumble)": generate_airplane_signature(fs, duration_sec),
        "Car (Broadband Noise)": generate_car_signature(fs, duration_sec)
    }

    for target_name, clean_sig in targets.items():
        clean_sig = clean_sig / (np.max(np.abs(clean_sig)) + 1e-8)
        clean_sig *= 10.0 # Standardize emission amplitude
        targets[target_name] = clean_sig

    distances = np.logspace(np.log10(100), np.log10(3000), 10)
    wind_levels = [0.0, 5.0, 15.0] # Clear, Moderate, Severe Gusts

    beamformers = {
        "GSC_MCWF": GSC_MCWF(fs=fs),
        "DAS": TaperedDAS(fs=fs)
    }

    # Nested dict for storing results to map
    results = {target_name: {w_level: {bf: [] for bf in beamformers.keys()} for w_level in wind_levels} for target_name in targets.keys()}

    for w_level in wind_levels:
        print(f"\n--- Wind Gust Level: {w_level} ---")

        # Generate wind noise profile
        noise = np.random.randn(50, N) * 0.005 # Baseline thermal/diffuse
        if w_level > 0:
            for m in range(50):
                gust_noise = np.random.randn(N)
                b, a = signal.butter(2, 50/(fs/2), btype='low')
                gust_noise = signal.filtfilt(b, a, gust_noise)
                env = np.zeros(N)
                start = int(N * 0.2)
                length = int(fs * 0.5)
                env[start:start+length] = np.hanning(length)
                noise[m] += gust_noise * env * w_level

        for target_name, clean_sig in targets.items():
            print(f"  Evaluating: {target_name}")

            for dist in distances:
                # Apply ISO 9613-1 absorption
                target_pos = np.array([0.0, dist, 50.0])
                target_multi_padded = apply_greens_function(clean_sig, fs, target_pos, pos, apply_iso=True)
                target_multi = target_multi_padded[:, :N]

                mixture = target_multi + noise

                for bf_name, bf in beamformers.items():
                    y_out = bf.process(mixture, pos, target_az=90.0, target_el=np.rad2deg(np.arcsin(50.0/dist)))
                    lam = evaluate_lam_preservation(clean_sig, y_out, fs)
                    results[target_name][w_level][bf_name].append(lam["embedding_similarity"])

    # Output Terminal Table
    print("\n### Max Detection Range Limits (LAM Trust >= 0.40) ###")
    for target_name in targets.keys():
        print(f"\n--- Target: {target_name} ---")
        print(f"{'Algorithm':<15} | {'Clear (Wind=0)':<15} | {'Mod (Wind=5)':<15} | {'Sev (Wind=15)':<15}")
        print("-" * 65)

        for bf_name in beamformers.keys():
            m_clear = np.max(distances[np.array(results[target_name][0.0][bf_name]) >= 0.40]) if any(np.array(results[target_name][0.0][bf_name]) >= 0.40) else 0
            m_mod = np.max(distances[np.array(results[target_name][5.0][bf_name]) >= 0.40]) if any(np.array(results[target_name][5.0][bf_name]) >= 0.40) else 0
            m_sev = np.max(distances[np.array(results[target_name][15.0][bf_name]) >= 0.40]) if any(np.array(results[target_name][15.0][bf_name]) >= 0.40) else 0
            print(f"{bf_name:<15} | {m_clear:>13.0f} m | {m_mod:>13.0f} m | {m_sev:>13.0f} m")

if __name__ == "__main__":
    run_2d_sweep()

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import sys

sys.path.append('.')
from simulator.acoustics.array import generate_log_spiral_array
from simulator.acoustics.channel import apply_greens_function
from simulator.beamformers import TaperedDAS, RobustLCMV, LCMV_SMI
from simulator.evaluation import evaluate_lam_preservation

def generate_drone_signature(fs, duration_sec):
    """ Generates a harmonic comb representing high-frequency rotor whine. """
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    f0 = 400.0 # Base rotor frequency
    sig = np.zeros_like(t)
    # Add 5 harmonics
    for i in range(1, 6):
        sig += np.sin(2 * np.pi * f0 * i * t) * (1.0 / i)
    return sig

def generate_airplane_signature(fs, duration_sec):
    """ Generates low-frequency 1/f rumble with blade-pass amplitude modulation. """
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    # 1/f noise
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 200/(fs/2), btype='low')
    rumble = signal.filtfilt(b, a, noise)

    # Blade pass modulation (e.g. 20 Hz)
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 20 * t))
    return rumble * modulation

def generate_car_signature(fs, duration_sec):
    """ Broadband tire/engine noise. """
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    noise = np.random.randn(len(t))
    b, a = signal.butter(4, [100/(fs/2), 1500/(fs/2)], btype='band')
    return signal.filtfilt(b, a, noise)

def run_pragmatic_sweep():
    print("==================================================")
    print("Executing Pragmatic Range Detectability Sweep")
    print("==================================================")

    fs = 16000
    duration_sec = 1.0
    N = int(fs * duration_sec)

    # 50-mic array
    pos = generate_log_spiral_array(n_sensors=50, aperture_m=2.5)

    # Target configurations
    targets = {
        "Drone (Rotor Whine)": generate_drone_signature(fs, duration_sec),
        "Propeller Airplane": generate_airplane_signature(fs, duration_sec),
        "Passing Car": generate_car_signature(fs, duration_sec)
    }

    # Distance sweep (logarithmic scale from 100m to 3000m)
    distances = np.logspace(np.log10(100), np.log10(3000), 10)

    beamformers = {
        "LCMV_Wideband": LCMV_SMI(fs=fs),
        "RobustLCMV_CMT": RobustLCMV(fs=fs, use_cmt=True),
        "DAS": TaperedDAS(fs=fs)
    }

    conditions = {
        "Clear": {"noise_std": 0.0005, "gust_amp": 0.0, "desc": "Low thermal noise, No wind"},
        "Windy": {"noise_std": 0.05, "gust_amp": 10.0, "desc": "High diffuse noise, Massive wind gusts"}
    }

    # Nested results dictionary
    results = {cond: {tgt: {bf: [] for bf in beamformers.keys()} for tgt in targets.keys()} for cond in conditions.keys()}
    results_single_mic = {cond: {tgt: [] for tgt in targets.keys()} for cond in conditions.keys()}

    # We set the baseline emission power. A Drone is normalized.
    for target_name, clean_sig in targets.items():
        clean_sig = clean_sig / (np.max(np.abs(clean_sig)) + 1e-8)
        clean_sig *= 10.0 # Standardize emission amplitude
        targets[target_name] = clean_sig

    for cond_name, env_cfg in conditions.items():
        print(f"\n==================================================")
        print(f"Running Condition: {cond_name.upper()} ({env_cfg['desc']})")

        # Generate the environmental noise once per condition
        noise = np.random.randn(50, N) * env_cfg["noise_std"]
        if env_cfg["gust_amp"] > 0:
            for m in range(50):
                gust_noise = np.random.randn(N)
                b, a = signal.butter(2, 50/(fs/2), btype='low')
                gust_noise = signal.filtfilt(b, a, gust_noise)
                # Random harsh envelope
                env = np.zeros(N)
                start = int(N * 0.2)
                length = int(fs * 0.5)
                env[start:start+length] = np.hanning(length)
                noise[m] += gust_noise * env * env_cfg["gust_amp"]

        for target_name, clean_sig in targets.items():
            print(f"  Evaluating: {target_name}")

            for dist in distances:
                target_pos = np.array([0.0, dist, 50.0]) # 50m altitude
                target_multi_padded = apply_greens_function(clean_sig, fs, target_pos, pos)
                target_multi = target_multi_padded[:, :N]

                mixture = target_multi + noise
                center_mic = mixture[0]

                lam_single = evaluate_lam_preservation(clean_sig, center_mic, fs)
                results_single_mic[cond_name][target_name].append(lam_single["embedding_similarity"])

                for bf_name, bf in beamformers.items():
                    # 1-degree look error injected to simulate slight real-world tracking mismatch
                    y_out = bf.process(mixture, pos, target_az=91.0, target_el=np.rad2deg(np.arcsin(50.0/dist)))
                    lam = evaluate_lam_preservation(clean_sig, y_out, fs)
                    results[cond_name][target_name][bf_name].append(lam["embedding_similarity"])

    # Terminal Output Table
    print("\n### Dual-Condition Range Limits (LAM Trust Threshold = 0.40) ###")
    for target_name in targets.keys():
        print(f"\n--- Target: {target_name} ---")
        print(f"{'Algorithm':<20} | {'Max Range (CLEAR)':<20} | {'Max Range (WINDY)':<20}")
        print("-" * 65)

        # Single Mic
        s_clear = np.array(results_single_mic["Clear"][target_name])
        s_windy = np.array(results_single_mic["Windy"][target_name])
        m_clear = np.max(distances[s_clear >= 0.40]) if any(s_clear >= 0.40) else 0
        m_windy = np.max(distances[s_windy >= 0.40]) if any(s_windy >= 0.40) else 0
        print(f"{'Single Mic':<20} | {m_clear:>15.0f} m | {m_windy:>15.0f} m")

        for bf_name in beamformers.keys():
            s_clear = np.array(results["Clear"][target_name][bf_name])
            s_windy = np.array(results["Windy"][target_name][bf_name])
            m_clear = np.max(distances[s_clear >= 0.40]) if any(s_clear >= 0.40) else 0
            m_windy = np.max(distances[s_windy >= 0.40]) if any(s_windy >= 0.40) else 0
            print(f"{bf_name:<20} | {m_clear:>15.0f} m | {m_windy:>15.0f} m")


if __name__ == "__main__":
    run_pragmatic_sweep()

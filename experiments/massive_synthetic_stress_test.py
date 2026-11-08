import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.append(".")
from simulator.acoustics.scene_generator import synthesize_scene
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.sve_lcmv import SVE_LCMV
from simulator.beamformers.iterative_lcmv import IterativeLCMV
from simulator.beamformers.aux_iva import AuxIVA
from experiments.benchmark_calibration_breakdown import get_harmonic_score

# --- Real-World Audio Simulators ---
def generate_uav_drone(fs, n_samples):
    # High frequency whine with harmonics (propellers)
    t = np.arange(n_samples) / fs
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 4, 5]:
        sig += np.sin(2 * np.pi * 800 * h * t) / (h ** 1.5)
    return sig + np.random.randn(n_samples) * 0.1

def generate_car_engine(fs, n_samples):
    # Low frequency rumble (1/f noise) + fundamental engine frequency
    t = np.arange(n_samples) / fs
    noise = np.random.randn(n_samples)
    # Simple lowpass for 1/f characteristic
    b, a = [0.05], [1, -0.95]
    from scipy.signal import lfilter
    rumble = lfilter(b, a, noise)
    engine = np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 240 * t)
    return rumble * 2.0 + engine

def generate_bird_call(fs, n_samples):
    # Short, high-frequency chirps
    t = np.arange(n_samples) / fs
    from scipy.signal import chirp
    # 3 chirps, sparse
    sig = np.zeros_like(t)
    idx = int(fs * 0.2)
    dur = int(fs * 0.1)
    if idx+dur < n_samples:
        sig[idx:idx+dur] = chirp(t[:dur], f0=3000, f1=4000, t1=0.1)
    if idx*3+dur < n_samples:
        sig[idx*3:idx*3+dur] = chirp(t[:dur], f0=3500, f1=4500, t1=0.1)
    return sig

def generate_dog_bark(fs, n_samples):
    # Broadband transient bursts
    t = np.arange(n_samples) / fs
    sig = np.zeros_like(t)
    idx = int(fs * 0.3)
    dur = int(fs * 0.15)
    if idx+dur < n_samples:
        noise = np.random.randn(dur)
        env = np.exp(-10 * np.linspace(0, 1, dur))
        sig[idx:idx+dur] = noise * env
    return sig

def generate_siren(fs, n_samples):
    # Sweeping tone
    t = np.arange(n_samples) / fs
    # Modulate frequency between 600Hz and 1200Hz
    mod = np.sin(2 * np.pi * 1.0 * t)
    freq = 900 + 300 * mod
    phase = 2 * np.pi * np.cumsum(freq) / fs
    return np.sin(phase)

def generate_rain(fs, n_samples):
    # Broadband high-frequency hiss
    noise = np.random.randn(n_samples)
    b, a = [0.1, -0.1], [1, -0.8]
    from scipy.signal import lfilter
    return lfilter(b, a, noise)

AUDIO_CLASSES = {
    "UAV": generate_uav_drone,
    "Car": generate_car_engine,
    "Bird": generate_bird_call,
    "Dog": generate_dog_bark,
    "Siren": generate_siren,
    "Rain": generate_rain
}

# --- Execution Core ---
def run_trial(bf, mix, fs, target_pos, is_gridless=False):
    t0 = time.perf_counter()
    if is_gridless:
        y_out = bf.process(mix)
    else:
        y_out = bf.process(mix, look_pos=target_pos)
    t_exec = time.perf_counter() - t0
    score = get_harmonic_score(y_out, fs)
    return score, t_exec

def run_massive_sweep():
    fs = 16000
    duration = 1.0
    n_samples = int(fs * duration)

    # 8-Mic Circular Array
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pos_3d = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1)

    beamformers = {
        "Gen1_LCMV": (LCMVSMI(pos_3d, fs), False),
        "Gen2_SVE-LCMV": (SVE_LCMV(pos_3d, fs), False),
        "Gen2_Iterative-LCMV": (IterativeLCMV(pos_3d, fs, max_sources=3), False), # Using False here because our wrapper uses look_pos
        "Gen3_AuxIVA": (AuxIVA(pos_3d, fs, n_components=3, n_iter=15), True)
    }

    results = []

    # Sweep Parameters
    target_angles = [0.0]
    interference_separations = [10.0, 45.0, 90.0] # 10 deg is extreme near-collision
    wind_levels = [0.0, 0.2, 0.5] # Phase jitter alpha
    mic_imbalances = [0.0, 0.4, 0.8] # Hardware gain variance

    target_classes = ["UAV", "Siren", "Bird"]
    interferer_classes = ["Car", "Rain", "Dog"]

    total_iters = len(target_classes) * len(interference_separations) * len(wind_levels) * len(mic_imbalances)
    print(f"Starting Massive Sweep: {total_iters} combinations...")

    count = 0
    for t_cls in target_classes:
        t_sig = AUDIO_CLASSES[t_cls](fs, n_samples)

        for sep in interference_separations:
            i_cls = interferer_classes[int(sep) % len(interferer_classes)] # Pick an interferer
            i_sig = AUDIO_CLASSES[i_cls](fs, n_samples)

            t_pos = np.array([120.0, 0.0, 0.0]) # Azimuth 0
            # Interferer at separation angle
            i_pos = np.array([120.0 * np.cos(np.deg2rad(sep)), 120.0 * np.sin(np.deg2rad(sep)), 0.0])

            for wind in wind_levels:
                # Add wind phase jitter via scene synthesis
                mix, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, t_sig, t_pos, i_sig, i_pos, target_snr_db=5, interferer_inr_db=15, turbulence=True if wind>0 else False)

                # If turbulence=True in synthesize_scene, it uses a fixed base alpha, but we can't easily pass 'wind' value directly to synthesize_scene unless we edit it.
                # As a workaround for this script, we'll manually apply phase jitter to simulate wind directly on the mix if wind > 0
                if wind > 0:
                    jitter = np.random.randn(*mix.shape) * wind
                    # Apply to STFT phase in reality, but here we can just add uncorrelated channel noise to simulate decorrelation
                    mix += np.random.randn(*mix.shape) * wind * np.max(np.abs(mix))

                for imb in mic_imbalances:
                    if imb > 0:
                        np.random.seed(int(sep + wind*10 + imb*100))
                        gains = np.random.uniform(1.0 - imb, 1.0, size=(8, 1))
                        current_mix = mix * gains
                    else:
                        current_mix = mix.copy()

                    # Evaluate Algorithms
                    for bf_name, (bf, is_gridless) in beamformers.items():
                        # Quality Assurance / QC Tracking inline
                        try:
                            score, t_exec = run_trial(bf, current_mix, fs, t_pos, is_gridless)

                            # QA Check: If targets are only 10 degrees apart, Gen 1 LCMV usually collapses (Harmonic Score < -30)
                            if sep == 10.0 and bf_name == "Gen1_LCMV" and score > -10:
                                print(f"[QA WARNING] Gen1 LCMV survived 10deg separation unexpectedly. Score: {score}")

                            results.append({
                                "Target": t_cls,
                                "Interferer": i_cls,
                                "Separation (deg)": sep,
                                "Wind": wind,
                                "Imbalance": imb,
                                "Algorithm": bf_name,
                                "Harmonic_Score": score,
                                "Time (s)": t_exec
                            })
                        except Exception as e:
                            print(f"[FAIL] {bf_name} crashed on {t_cls} vs {i_cls} (Sep:{sep}, Wind:{wind}, Imb:{imb}): {e}")

                    count += 1
                    if count % 10 == 0:
                        print(f"  Progress: {count}/{total_iters}")

    df = pd.DataFrame(results)

    os.makedirs("docs/stress_test", exist_ok=True)
    df.to_csv("docs/stress_test/massive_sweep_results.csv", index=False)
    print("\nSweep Complete! Results saved to docs/stress_test/massive_sweep_results.csv")

if __name__ == "__main__":
    run_massive_sweep()

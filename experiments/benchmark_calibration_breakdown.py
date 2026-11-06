import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(".")
from acoustic_sim.dsp_array import load_array
from simulator.acoustics.scene_generator import synthesize_scene
from experiments.feature_extraction.signal_processing import compute_spectrogram_and_peaks
from experiments.feature_extraction.harmonic_detection import track_harmonics

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.robust_socp import RobustSOCP
from simulator.beamformers.ebb import EBB
from simulator.beamformers.gev import GEV

def generate_tonal_signal(fs, n_samples, f0=400, harmonics=5):
    t = np.arange(n_samples) / fs
    sig = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        sig += np.sin(2 * np.pi * f0 * h * t) / h
    return sig

def get_harmonic_score(audio, fs):
    f, t, Pxx_db, peaks = compute_spectrogram_and_peaks(audio, fs, parallel=False)
    tracks = track_harmonics(peaks)
    if tracks:
        return float(tracks[0]['max_score'])
    return -150.0

def main():
    fs = 16000
    duration = 2.0
    n_samples = int(fs * duration)

    # Setup Geometry (assuming circular for benchmark)
    M = 16
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(M)
    pos_3d = np.stack([x, y, z], axis=1)

    # Instantiate Beamformers
    bfs = {
        "DAS": StandardDAS(pos_3d, fs),
        "LCMV": LCMVSMI(pos_3d, fs),
        "PhaseOnlyLCMV": PhaseOnlyLCMV(pos_3d, fs),
        "RobustSOCP": RobustSOCP(pos_3d, fs),
        "EBB": EBB(pos_3d, fs),
        "GEV": GEV(pos_3d, fs)
    }

    # Sweep Parameters
    variance_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] # From perfect to extreme mismatch
    target_pos = np.array([120.0, 0.0, 0.0]) # Azimuth 0
    interferer_pos = np.array([0.0, 20.0, 0.0]) # Azimuth 90

    target_sig = generate_tonal_signal(fs, n_samples, f0=400, harmonics=5)
    int_sig = np.random.randn(n_samples) # broadband noise interferer

    results = []

    print("Starting Calibration Breakdown Sweep...")
    for var in variance_levels:
        print(f"Testing Gain Variance Level: {var*100}%")

        # Synthesize Scene
        mix, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, target_sig, target_pos, int_sig, interferer_pos,
                                        target_snr_db=5, interferer_inr_db=20)

        # Apply Hardware Miscalibration
        if var > 0:
            np.random.seed(42)
            # gains center at 1.0, range [1-var, 1+var]
            gains = np.random.uniform(1.0 - var, 1.0 + var, size=(M, 1))
            mix = mix * gains

        # Process each beamformer
        for name, bf in bfs.items():
            try:
                y_out = bf.process(mix, look_pos=target_pos)
                score = get_harmonic_score(y_out, fs)
            except Exception as e:
                print(f"Error {name}: {e}")
                score = -150.0

            results.append({
                "Variance": var,
                "Algorithm": name,
                "Harmonic_Score": score
            })

    df = pd.DataFrame(results)

    # Plot results
    plt.figure(figsize=(10, 6))
    for name in bfs.keys():
        sub_df = df[df["Algorithm"] == name]
        plt.plot(sub_df["Variance"] * 100, sub_df["Harmonic_Score"], marker='o', label=name, linewidth=2)

    plt.title("Beamformer Robustness vs. Uncalibrated Microphones", fontsize=14)
    plt.xlabel("Microphone Gain Variance Error (%)", fontsize=12)
    plt.ylabel("Target Harmonic Detectability Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs("docs/figures", exist_ok=True)
    plt.savefig("docs/figures/calibration_breakdown.png")

    print("\nBenchmark Complete!")
    print(df.pivot(index="Variance", columns="Algorithm", values="Harmonic_Score").to_markdown())

if __name__ == "__main__":
    main()

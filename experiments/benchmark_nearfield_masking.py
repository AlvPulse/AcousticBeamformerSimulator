import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.append(".")
from simulator.acoustics.scene_generator import synthesize_scene
from experiments.benchmark_calibration_breakdown import get_harmonic_score
from experiments.massive_synthetic_stress_test import AUDIO_CLASSES
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.sve_lcmv import SVE_LCMV
from simulator.beamformers.iterative_lcmv import IterativeLCMV
from simulator.beamformers.aux_iva import AuxIVA

def run_nearfield_trial(bf_name, bf, mix, fs, target_pos, is_gridless=False):
    t0 = time.perf_counter()
    if is_gridless:
        y_out = bf.process(mix)
    else:
        y_out = bf.process(mix, look_pos=target_pos)
    t_exec = time.perf_counter() - t0

    # Calculate IRR to measure how much the near-field interferer bled through
    # Target is weak, interferer is massive.
    # IRR = var(output) / var(mix - output).
    # High IRR = we successfully extracted the weak target and rejected the massive interferer.

    # Ensure lengths match due to ISTFT framing overlap
    min_len = min(len(y_out), len(mix[0]))
    y_out_trunc = y_out[:min_len]
    mix_trunc = mix[0, :min_len]

    irr_db = 10 * np.log10(np.var(y_out_trunc) / (np.var(mix_trunc - y_out_trunc) + 1e-12))
    score = get_harmonic_score(y_out, fs)

    return score, irr_db, t_exec

def run_nearfield_masking_benchmark():
    fs = 16000
    duration = 1.0
    n_samples = int(fs * duration)

    # Standard 8-Mic Circular Array
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pos_3d = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1)

    beamformers = {
        "Gen1_LCMV": (LCMVSMI(pos_3d, fs), False),
        "Gen2_SVE-LCMV": (SVE_LCMV(pos_3d, fs), False),
        "Gen3_AuxIVA": (AuxIVA(pos_3d, fs, n_components=2, n_iter=15), True)
    }

    results = []

    # --- Scenario Parameters ---
    # Weak Far-Field Target (e.g. UAV at 1500m)
    target_pos = np.array([1500.0, 0.0, 0.0]) # Azimuth 0
    target_sig = AUDIO_CLASSES["UAV"](fs, n_samples)
    target_snr_db = -15.0 # Very weak

    # Massive Near-Field Interferers (e.g. Bird/Bark at 1m-5m)
    interferer_classes = ["Bird", "Dog", "Car"]
    near_field_distances = [1.0, 3.0, 5.0]
    interferer_inr_db = 35.0 # Massive masking interference

    wind_levels = [0.0, 0.3] # Test Phase Jitter coupling with spherical mismatch

    print("Starting Near-Field Masking Benchmark...")

    for i_cls in interferer_classes:
        i_sig = AUDIO_CLASSES[i_cls](fs, n_samples)

        for dist in near_field_distances:
            # Interferer at Azimuth 90, strictly in the Near-Field
            i_pos = np.array([0.0, dist, 0.0])

            for wind in wind_levels:

                # Synthesize the scene.
                # synthesize_scene automatically computes exact spherical delays since target_pos and i_pos are 3D coordinates.
                mix, t_mc, i_mc, _ = synthesize_scene(
                    pos_3d, fs, n_samples,
                    target_sig, target_pos,
                    i_sig, i_pos,
                    target_snr_db=target_snr_db, interferer_inr_db=interferer_inr_db,
                    turbulence=True if wind > 0 else False
                )

                # Apply explicit wind phase jitter
                if wind > 0:
                    mix += np.random.randn(*mix.shape) * wind * np.max(np.abs(mix))

                # Evaluate Algorithms
                for bf_name, (bf, is_gridless) in beamformers.items():
                    try:
                        score, irr_db, t_exec = run_nearfield_trial(bf_name, bf, mix, fs, target_pos, is_gridless)
                        results.append({
                            "Target": "Weak Far-Field UAV",
                            "Near-Field Interferer": i_cls,
                            "NF Distance (m)": dist,
                            "Wind Level": wind,
                            "Algorithm": bf_name,
                            "Harmonic_Score": score,
                            "Interference_Rejection_IRR_dB": irr_db,
                            "Execution Time (s)": t_exec
                        })
                    except Exception as e:
                        print(f"Failed {bf_name} on {i_cls} dist={dist}: {e}")

    df = pd.DataFrame(results)

    os.makedirs("docs/stress_test", exist_ok=True)
    df.to_csv("docs/stress_test/nearfield_masking_results.csv", index=False)

    print("\n--- NEAR-FIELD MASKING RESULTS SUMMARY ---")
    summary = df.groupby(["Near-Field Interferer", "Algorithm"])[["Harmonic_Score", "Interference_Rejection_IRR_dB"]].mean()
    print(summary.to_markdown())
    print("\nBenchmark Complete! Results saved to docs/stress_test/nearfield_masking_results.csv")

if __name__ == "__main__":
    run_nearfield_masking_benchmark()

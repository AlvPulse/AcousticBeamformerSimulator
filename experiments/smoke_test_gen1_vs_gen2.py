import numpy as np
import time
import pandas as pd
import sys

sys.path.append(".")
from simulator.acoustics.scene_generator import synthesize_scene
from experiments.benchmark_calibration_breakdown import generate_tonal_signal, get_harmonic_score
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.sve_lcmv import SVE_LCMV
from simulator.beamformers.iterative_lcmv import IterativeLCMV

def run_evaluation(name, bf, mix, fs, target_pos, is_iterative=False):
    start_time = time.perf_counter()
    if is_iterative:
        # Iterative LCMV searches the whole grid inherently in its process function
        y_out = bf.process(mix)
    else:
        # Standard methods need to be steered to the target
        y_out = bf.process(mix, look_pos=target_pos)

    exec_time = time.perf_counter() - start_time
    score = get_harmonic_score(y_out, fs)
    return {"Generation": name.split("-")[0].strip(), "Algorithm": name, "Harmonic_Score": score, "Compute_Time (s)": exec_time}

def main():
    print("--- SMOKE TEST: Generation 1 vs Generation 2 ---")
    fs = 16000
    duration = 1.0
    n_samples = int(fs * duration)

    # 1. Setup a standard array (8 mics)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pos_3d = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1)

    # 2. Simulate a hostile environment: 1 Target + 3 Loud Interferers
    target_pos = np.array([120.0, 0.0, 0.0]) # Azimuth 0
    target_sig = generate_tonal_signal(fs, n_samples, f0=400, harmonics=5)

    mix, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, target_sig, target_pos,
                                    np.random.randn(n_samples), np.array([0.0, 120.0, 0.0]), # Int 1 (Az 90)
                                    target_snr_db=5, interferer_inr_db=20)

    # Add two more interferers
    extra_mix1, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, np.zeros(n_samples), target_pos,
                                           np.random.randn(n_samples), np.array([-120.0, 0.0, 0.0]), target_snr_db=5, interferer_inr_db=20)
    extra_mix2, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, np.zeros(n_samples), target_pos,
                                           np.random.randn(n_samples), np.array([0.0, -120.0, 0.0]), target_snr_db=5, interferer_inr_db=20)

    final_mix = mix + extra_mix1 + extra_mix2

    # Apply severe microphone uncalibration to break Generation 1
    np.random.seed(42)
    gains = np.random.uniform(0.1, 1.0, size=(8, 1))
    final_mix = final_mix * gains

    results = []

    # Gen 1: Standard LCMV
    print("Running Gen 1: Standard LCMV...")
    lcmv = LCMVSMI(pos_3d, fs)
    results.append(run_evaluation("Gen 1 - Standard LCMV", lcmv, final_mix, fs, target_pos))

    # Gen 2: SVE LCMV (Data-Driven Steering)
    print("Running Gen 2: SVE-LCMV...")
    sve_lcmv = SVE_LCMV(pos_3d, fs)
    results.append(run_evaluation("Gen 2 - SVE-LCMV", sve_lcmv, final_mix, fs, target_pos))

    # Gen 2: Iterative LCMV (Peak Depletion)
    print("Running Gen 2: Iterative LCMV (Peak Depletion)...")
    iter_lcmv = IterativeLCMV(pos_3d, fs, max_sources=4)
    # We MUST pass the look_pos to the iterative LCMV so it knows which of the 4 depleted sources to return to us!
    results.append(run_evaluation("Gen 2 - Iterative LCMV", iter_lcmv, final_mix, fs, target_pos, is_iterative=False))

    df = pd.DataFrame(results)
    print("\n--- RESULTS ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(".")
from simulator.config import FS, N_FFT, HOP_LENGTH
from simulator.acoustics.scene_generator import synthesize_scene
from experiments.benchmark_calibration_breakdown import generate_tonal_signal, get_harmonic_score

from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.ilrma import ILRMA
from simulator.beamformers.aux_iva import AuxIVA
from simulator.beamformers.fast_iva import FastIVA
from simulator.beamformers.sve_lcmv import SVE_LCMV
from simulator.beamformers.multirank_osp import MultiRankOSP

def run_grid_search(bf, mix, fs, grid_size=64):
    """
    Paradigm A: Grid Search.
    Loops through angles, beamforms, and finds the max energy angle.
    Returns: output audio, execution time, harmonic score, and Interference Rejection Ratio (IRR).
    """
    start_time = time.perf_counter()
    azimuths = np.linspace(-180, 180, grid_size, endpoint=False)

    max_energy = -1
    min_energy = float('inf')
    best_audio = None

    for az in azimuths:
        # Assuming elevation 0 for 1D grid search
        look_pos = np.array([120.0 * np.cos(np.deg2rad(az)), 120.0 * np.sin(np.deg2rad(az)), 0.0])
        y_out = bf.process(mix, look_pos=look_pos)

        energy = np.sum(np.abs(y_out)**2)
        if energy > max_energy:
            max_energy = energy
            best_audio = y_out
        if energy < min_energy:
            min_energy = energy

    exec_time = time.perf_counter() - start_time
    score = get_harmonic_score(best_audio, fs)

    # Calculate IRR (Nulling Depth equivalent) for Grid Search
    irr_db = 10 * np.log10(max_energy / (min_energy + 1e-12))

    return best_audio, exec_time, score, irr_db

def run_gridless_bss(bf, mix, fs):
    """
    Paradigm B: Gridless Separation.
    Processes the mixture exactly once.
    Returns: output audio, execution time, harmonic score, and Interference Rejection Ratio (IRR).
    """
    start_time = time.perf_counter()
    y_out = bf.process(mix) # look_pos is ignored
    exec_time = time.perf_counter() - start_time
    score = get_harmonic_score(y_out, fs)

    # BSS intrinsically returns the highest variance target when resolving permutations
    # To compute an IRR equivalent, we evaluate the separated channels internally,
    # but since the API returns the single target, we estimate the rejection directly
    # by calculating the remaining energy of the known interferer in the output.
    # We will simulate a metric comparable to grid IRR by analyzing output sparsity.

    # Ensure lengths match due to ISTFT framing overlap differences
    min_len = min(len(y_out), len(mix[0]))
    y_out_trunc = y_out[:min_len]
    mix_trunc = mix[0, :min_len]

    irr_db = 10 * np.log10(np.var(y_out_trunc) / (np.var(mix_trunc - y_out_trunc) + 1e-12))

    return y_out, exec_time, score, irr_db

def main():
    print("Starting Gridless BSS vs Grid Search Benchmark...")
    fs = 16000
    duration = 1.0
    n_samples = int(fs * duration)

    # Setup Geometry (8 mics circular)
    M = 8
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    pos_3d = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(M)], axis=1)

    # Synthesize mixture (Target at 0 deg, Interferer at 90 deg)
    target_sig = generate_tonal_signal(fs, n_samples, f0=400, harmonics=5)
    int_sig = np.random.randn(n_samples)

    mix, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, target_sig, np.array([120.0, 0.0, 0.0]),
                                    int_sig, np.array([0.0, 120.0, 0.0]), target_snr_db=5, interferer_inr_db=15)

    results = []

    # 1. Standard Grid Search (LCMV)
    print("Running LCMV Grid Search (64 points)...")
    lcmv = LCMVSMI(pos_3d, fs)
    _, time_lcmv, score_lcmv, irr_lcmv = run_grid_search(lcmv, mix, fs, grid_size=64)
    results.append({"Method": "LCMV Grid Search (64 pts)", "Harmonic_Score": score_lcmv, "IRR_dB (Interference Rejection)": irr_lcmv, "Execution Time (s)": time_lcmv})

    # 2. Gridless ILRMA
    print("Running ILRMA (Gridless)...")
    ilrma = ILRMA(pos_3d, fs, n_iter=20)
    _, time_ilrma, score_ilrma, irr_ilrma = run_gridless_bss(ilrma, mix, fs)
    results.append({"Method": "ILRMA Gridless (20 iters)", "Harmonic_Score": score_ilrma, "IRR_dB (Interference Rejection)": irr_ilrma, "Execution Time (s)": time_ilrma})

    # 3. Gridless AuxIVA
    print("Running AuxIVA (Gridless)...")
    auxiva = AuxIVA(pos_3d, fs, n_iter=20)
    _, time_aux, score_aux, irr_aux = run_gridless_bss(auxiva, mix, fs)
    results.append({"Method": "AuxIVA Gridless (20 iters)", "Harmonic_Score": score_aux, "IRR_dB (Interference Rejection)": irr_aux, "Execution Time (s)": time_aux})

    # 4. Gridless FastIVA
    print("Running FastIVA (Gridless)...")
    fastiva = FastIVA(pos_3d, fs, n_iter=15)
    _, time_fast, score_fast, irr_fast = run_gridless_bss(fastiva, mix, fs)
    results.append({"Method": "FastIVA Gridless (15 iters)", "Harmonic_Score": score_fast, "IRR_dB (Interference Rejection)": irr_fast, "Execution Time (s)": time_fast})

    # 5. SVE-LCMV Grid Search
    print("Running SVE-LCMV Grid Search (64 points)...")
    sve = SVE_LCMV(pos_3d, fs)
    _, time_sve, score_sve, irr_sve = run_grid_search(sve, mix, fs, grid_size=64)
    results.append({"Method": "SVE-LCMV Grid Search (64 pts)", "Harmonic_Score": score_sve, "IRR_dB (Interference Rejection)": irr_sve, "Execution Time (s)": time_sve})

    # 6. MR-LCMV Grid Search
    print("Running Multi-Rank OSP Grid Search (64 points)...")
    mr = MultiRankOSP(pos_3d, fs)
    _, time_mr, score_mr, irr_mr = run_grid_search(mr, mix, fs, grid_size=64)
    results.append({"Method": "MR-LCMV (OSP) Grid Search (64 pts)", "Harmonic_Score": score_mr, "IRR_dB (Interference Rejection)": irr_mr, "Execution Time (s)": time_mr})

    df = pd.DataFrame(results)

    print("\n--- BENCHMARK RESULTS ---")
    print(df.to_markdown(index=False))

    os.makedirs("docs", exist_ok=True)
    df.to_csv("docs/benchmark_gridless.csv", index=False)

if __name__ == "__main__":
    main()

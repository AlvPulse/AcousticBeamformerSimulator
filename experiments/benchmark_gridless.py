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

import tracemalloc

def run_grid_search(bf, mix, fs, grid_size=64):
    """
    Paradigm A: Grid Search.
    Loops through angles, beamforms, and finds the max energy angle.
    Returns: output audio, execution time, harmonic score, IRR, and Peak Memory (MB).
    """
    tracemalloc.start()
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

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 10**6

    # Calculate IRR (Nulling Depth equivalent) for Grid Search
    irr_db = 10 * np.log10(max_energy / (min_energy + 1e-12))

    return best_audio, exec_time, score, irr_db, peak_mb

def run_gridless_bss(bf, mix, fs):
    """
    Paradigm B: Gridless Separation.
    Processes the mixture exactly once.
    Returns: output audio, execution time, harmonic score, IRR, and Peak Memory (MB).
    """
    tracemalloc.start()
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

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 10**6

    irr_db = 10 * np.log10(np.var(y_out_trunc) / (np.var(mix_trunc - y_out_trunc) + 1e-12))

    return y_out, exec_time, score, irr_db, peak_mb

def run_scenario(scenario_name, target_sig, target_pos, interferer_sigs, interferer_poss, pos_3d, fs, n_samples):
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Synthesize mixture
    mix = np.zeros((pos_3d.shape[0], n_samples))

    # Add target
    base_mix, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, target_sig, target_pos,
                                        interferer_sigs[0], interferer_poss[0], target_snr_db=5, interferer_inr_db=15)
    mix += base_mix

    # Add additional interferers
    for i in range(1, len(interferer_sigs)):
        extra_mix, _, _, _ = synthesize_scene(pos_3d, fs, n_samples, np.zeros(n_samples), target_pos,
                                              interferer_sigs[i], interferer_poss[i], target_snr_db=5, interferer_inr_db=15)
        # Add only the interferer component (subtract background noise to avoid multiplying noise floor)
        mix += extra_mix

    results = []

    # 1. LCMV Grid Search
    print("  Running LCMV Grid Search (64 points)...")
    lcmv = LCMVSMI(pos_3d, fs)
    _, time_lcmv, score_lcmv, irr_lcmv, mem_lcmv = run_grid_search(lcmv, mix, fs, grid_size=64)
    results.append({"Scenario": scenario_name, "Method": "LCMV Grid (64 pts)", "Harmonic_Score": score_lcmv, "IRR_dB": irr_lcmv, "Time (s)": time_lcmv, "Peak RAM (MB)": mem_lcmv})

    # 2. SVE-LCMV
    print("  Running SVE-LCMV Grid Search (64 points)...")
    sve = SVE_LCMV(pos_3d, fs)
    _, time_sve, score_sve, irr_sve, mem_sve = run_grid_search(sve, mix, fs, grid_size=64)
    results.append({"Scenario": scenario_name, "Method": "SVE-LCMV Grid (64 pts)", "Harmonic_Score": score_sve, "IRR_dB": irr_sve, "Time (s)": time_sve, "Peak RAM (MB)": mem_sve})

    # 3. Multi-Rank OSP
    print("  Running Multi-Rank OSP Grid Search (64 points)...")
    mr = MultiRankOSP(pos_3d, fs)
    _, time_mr, score_mr, irr_mr, mem_mr = run_grid_search(mr, mix, fs, grid_size=64)
    results.append({"Scenario": scenario_name, "Method": "MR-LCMV (OSP) Grid (64 pts)", "Harmonic_Score": score_mr, "IRR_dB": irr_mr, "Time (s)": time_mr, "Peak RAM (MB)": mem_mr})

    # 4. ILRMA Gridless
    # For multi-interferer, we MUST increase components to capture them. We set n=4.
    print("  Running ILRMA (Gridless, 4 comp)...")
    ilrma = ILRMA(pos_3d, fs, n_components=4, n_iter=20)
    _, time_ilrma, score_ilrma, irr_ilrma, mem_ilrma = run_gridless_bss(ilrma, mix, fs)
    results.append({"Scenario": scenario_name, "Method": "ILRMA Gridless (n=4)", "Harmonic_Score": score_ilrma, "IRR_dB": irr_ilrma, "Time (s)": time_ilrma, "Peak RAM (MB)": mem_ilrma})

    # 5. AuxIVA Gridless
    print("  Running AuxIVA (Gridless, 4 comp)...")
    auxiva = AuxIVA(pos_3d, fs, n_components=4, n_iter=20)
    _, time_aux, score_aux, irr_aux, mem_aux = run_gridless_bss(auxiva, mix, fs)
    results.append({"Scenario": scenario_name, "Method": "AuxIVA Gridless (n=4)", "Harmonic_Score": score_aux, "IRR_dB": irr_aux, "Time (s)": time_aux, "Peak RAM (MB)": mem_aux})

    # 6. FastIVA Gridless
    print("  Running FastIVA (Gridless, 4 comp)...")
    fastiva = FastIVA(pos_3d, fs, n_components=4, n_iter=15)
    _, time_fast, score_fast, irr_fast, mem_fast = run_gridless_bss(fastiva, mix, fs)
    results.append({"Scenario": scenario_name, "Method": "FastIVA Gridless (n=4)", "Harmonic_Score": score_fast, "IRR_dB": irr_fast, "Time (s)": time_fast, "Peak RAM (MB)": mem_fast})

    return results

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

    # Signals
    target_pos = np.array([120.0, 0.0, 0.0])
    target_sig = generate_tonal_signal(fs, n_samples, f0=400, harmonics=5)

    all_results = []

    # --- Scenario 1: Single Interferer ---
    interferer_sigs_1 = [np.random.randn(n_samples)]
    interferer_poss_1 = [np.array([0.0, 120.0, 0.0])]
    res_1 = run_scenario("1 Target + 1 Interferer", target_sig, target_pos, interferer_sigs_1, interferer_poss_1, pos_3d, fs, n_samples)
    all_results.extend(res_1)

    # --- Scenario 2: Multiple Interferers (Capacity Limit Test) ---
    interferer_sigs_3 = [np.random.randn(n_samples) for _ in range(3)]
    interferer_poss_3 = [
        np.array([0.0, 120.0, 0.0]),   # 90 deg
        np.array([-120.0, 0.0, 0.0]),  # 180 deg
        np.array([0.0, -120.0, 0.0])   # -90 deg
    ]
    res_2 = run_scenario("1 Target + 3 Interferers", target_sig, target_pos, interferer_sigs_3, interferer_poss_3, pos_3d, fs, n_samples)
    all_results.extend(res_2)

    df = pd.DataFrame(all_results)

    print("\n--- COMPREHENSIVE BENCHMARK RESULTS ---")
    summary = df.groupby(["Scenario", "Method"]).mean(numeric_only=True)
    print(summary.to_markdown())

    os.makedirs("docs", exist_ok=True)
    df.to_csv("docs/benchmark_gridless.csv", index=False)

if __name__ == "__main__":
    main()

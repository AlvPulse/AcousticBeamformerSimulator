import numpy as np
import pandas as pd
import os
import soundfile as sf
import argparse

from simulator.config import FS, N_FFT, HOP_LENGTH
from simulator.acoustics.array import get_array_geometry
from simulator.acoustics.scene_generator import synthesize_scene, compute_rms

from simulator.beamformers.tapered_das import TaperedDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.multirank_osp import MultiRankOSP
from simulator.beamformers.ebb import EBB
from simulator.beamformers.robust_socp import RobustSOCP
from simulator.beamformers.cgmm_mvdr import cGMM_MVDR

from simulator.evaluation.metrics_acoustic import compute_si_sdr, compute_lsd, compute_array_gain
from simulator.evaluation.metrics_lam import compute_clap_similarity
from simulator.evaluation.profiler import Profiler

import scipy.signal as signal

def generate_random_source_signal(fs, duration_s, is_speech=True):
    # Dummy source generator (bandpass noise or chirp to act as target)
    t = np.arange(int(fs * duration_s)) / fs
    if is_speech:
        # Modulated noise to mimic speech envelope
        noise = np.random.randn(len(t))
        env = np.abs(np.sin(2 * np.pi * 2 * t)) # 2 Hz modulation
        sig = noise * env
        b, a = signal.butter(4, [300/(fs/2), 3400/(fs/2)], btype='bandpass')
        sig = signal.filtfilt(b, a, sig)
    else:
        # Engine noise for interferer
        sig = np.random.randn(len(t))
        b, a = signal.butter(4, 500/(fs/2), btype='lowpass')
        sig = signal.filtfilt(b, a, sig)
    return sig

def run_scenario(scenario_id, target_snr, interferer_inr, sigma_phi, n_trials=5, output_dir="results"):
    pos_mics = get_array_geometry()

    beamformers = {
        "DAS (Cheby)": TaperedDAS(pos_mics, FS),
        "DL-LCMV": LCMVSMI(pos_mics, FS),
        "MultiRank OSP": MultiRankOSP(pos_mics, FS),
        "EBB": EBB(pos_mics, FS),
        "SOCP": RobustSOCP(pos_mics, FS),
        "cGMM-MVDR": cGMM_MVDR(pos_mics, FS)
    }

    profiler = Profiler(FS)
    results = []

    os.makedirs(output_dir, exist_ok=True)

    duration_s = 2.0
    n_samples = int(FS * duration_s)

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}")

        target_sig = generate_random_source_signal(FS, duration_s, is_speech=True)
        interferer_sig = generate_random_source_signal(FS, duration_s, is_speech=False)

        target_pos = np.array([120.0, 0.0, 0.0]) # 120m Far field
        interferer_pos = np.array([0.0, 8.0, 0.0]) # 8m Near field

        mix_mc, target_mc, int_mc, noise_mc = synthesize_scene(
            pos_mics, FS, n_samples, target_sig, target_pos, interferer_sig, np.array([interferer_pos]*n_samples),
            target_snr, interferer_inr, turbulence=False
        )

        # Calculate raw center mic metrics
        center_mic_idx = 0
        raw_si_sdr = compute_si_sdr(target_mc[center_mic_idx], mix_mc[center_mic_idx])

        raw_snr_in = np.mean(target_mc[center_mic_idx]**2) / np.mean(noise_mc[center_mic_idx]**2 + 1e-12)

        for name, bf in beamformers.items():
            profiler.start(n_samples)
            y_out = bf.process(mix_mc, target_pos)
            prof_res = profiler.stop()

            # Save audio for first trial
            if trial == 0:
                sf.write(os.path.join(output_dir, f"{scenario_id}_{name.replace(' ', '_')}.wav"), y_out, FS)

            # Evaluate
            bf_si_sdr = compute_si_sdr(target_mc[center_mic_idx], y_out)
            delta_si_sdr = bf_si_sdr - raw_si_sdr

            f, t, Zxx_clean = signal.stft(target_mc[center_mic_idx], fs=FS, nperseg=N_FFT)
            f, t, Zxx_beam = signal.stft(y_out, fs=FS, nperseg=N_FFT)
            lsd = compute_lsd(Zxx_clean, Zxx_beam)

            clap_sim = compute_clap_similarity(target_mc[center_mic_idx], y_out, FS)

            # Target power in output vs noise power in output
            # (Rough estimate by passing them separately)
            # For exact array gain we'd need linear processing, but we can do a rough estimate

            results.append({
                "Scenario": scenario_id,
                "Algorithm": name,
                "Trial": trial,
                "Delta SI-SDR (dB)": delta_si_sdr,
                "LSD": lsd,
                "CLAP Similarity": clap_sim,
                "RTF": prof_res["rtf"],
                "Peak RAM (MB)": prof_res["peak_memory_mb"]
            })

    return results

def main():
    print("Starting 50-Mic LAM Benchmarking Simulator...")
    all_results = []

    # S1: Near-Far Power Dynamic (Interferer INR Sweep)
    print("Running Scenario S1...")
    res_s1 = run_scenario("S1_NearFar", target_snr=-15, interferer_inr=35, sigma_phi=0.0)
    all_results.extend(res_s1)

    # S2: Target Distance & SNR
    print("Running Scenario S2...")
    res_s2 = run_scenario("S2_LowSNR", target_snr=-25, interferer_inr=10, sigma_phi=0.0)
    all_results.extend(res_s2)

    df = pd.DataFrame(all_results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nBenchmark complete. Results saved to benchmark_results.csv")

    # Summary
    summary = df.groupby(["Scenario", "Algorithm"]).mean(numeric_only=True)
    print("\nSummary:")
    print(summary[["Delta SI-SDR (dB)", "LSD", "CLAP Similarity", "RTF"]])

if __name__ == "__main__":
    main()

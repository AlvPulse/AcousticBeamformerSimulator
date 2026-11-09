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
from simulator.beamformers.ilrma import ILRMA
from simulator.beamformers.fast_iva import FastIVA
from simulator.beamformers.gev import GEV
from simulator.beamformers.multirank_osp import MultiRankOSP
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.music import MUSIC
from simulator.beamformers.srp_phat import SRP_PHAT
from simulator.beamformers.incm_mvdr import INCM_MVDR
from simulator.beamformers.das import StandardDAS
from simulator.beamformers.tapered_das import TaperedDAS
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.robust_socp import RobustSOCP
from simulator.beamformers.ebb import EBB
from simulator.beamformers.cgmm_mvdr import cGMM_MVDR

def calculate_fraunhofer_distance(pos_mics, max_freq=4000.0, c=343.0):
    """
    Calculates the boundary between Near-Field and Far-Field (Fraunhofer Distance).
    d = 2 * D^2 / lambda, where D is the max aperture of the array.
    """
    # Calculate maximum diameter D
    distances = np.linalg.norm(pos_mics[:, None, :] - pos_mics[None, :, :], axis=2)
    D = np.max(distances)
    wavelength = c / max_freq
    d_f = (2 * (D ** 2)) / wavelength
    return d_f, D

def run_nearfield_trial(bf_name, bf, mix, fs, target_pos, is_gridless=False):
    t0 = time.perf_counter()
    if is_gridless:
        y_out = bf.process(mix)
    else:
        y_out = bf.process(mix, look_pos=target_pos)
    t_exec = time.perf_counter() - t0

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
    radius = 1.0 # 2-meter diameter array
    pos_3d = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(8)], axis=1)

    fraunhofer, array_diameter = calculate_fraunhofer_distance(pos_3d, max_freq=4000.0)
    print(f"Array Diameter: {array_diameter:.2f} m")
    print(f"Fraunhofer Near-Field Boundary (at 4000Hz): {fraunhofer:.2f} m")
    print("Any source closer than this distance will cause spherical plane-wave distortion.")

    beamformers = {
        "Standard DAS": (StandardDAS(pos_3d, fs), False),
        "Tapered_DAS": (TaperedDAS(pos_3d, fs), False),
        "Gen1_LCMV": (LCMVSMI(pos_3d, fs), False),
        "MUSIC": (MUSIC(pos_3d, fs), False),
        "SRP_PHAT": (SRP_PHAT(pos_3d, fs), False),
        "PhaseOnlyLCMV": (PhaseOnlyLCMV(pos_3d, fs), False),
        "RobustLCMV": (RobustLCMV(pos_3d, fs), False),
        "RobustSOCP": (RobustSOCP(pos_3d, fs), False),
        "EBB": (EBB(pos_3d, fs), False),
        "cGMM_MVDR": (cGMM_MVDR(pos_3d, fs), False),
        "GSC_MCWF": (GSC_MCWF(pos_3d, fs), False),
        "MultiRankOSP": (MultiRankOSP(pos_3d, fs), False),
        "SVE_LCMV": (SVE_LCMV(pos_3d, fs), False),
        "INCM_MVDR": (INCM_MVDR(pos_3d, fs), False),
        "Iterative_LCMV": (IterativeLCMV(pos_3d, fs, max_sources=3), False),
        "AuxIVA": (AuxIVA(pos_3d, fs, n_components=2, n_iter=15), True),
        "FastIVA": (FastIVA(pos_3d, fs, n_components=2, n_iter=15), True),
        "ILRMA": (ILRMA(pos_3d, fs, n_components=2, n_iter=15), True),
        "GEV": (GEV(pos_3d, fs), True)
    }

    results = []

    # --- Scenario Parameters ---
    target_pos = np.array([1500.0, 0.0, 0.0]) # Azimuth 0
    target_sig = AUDIO_CLASSES["UAV"](fs, n_samples)
    target_snr_db = -10.0 # Very weak

    # Massive Near-Field Interferers (e.g. Bird/Bark) inside the Fraunhofer distance
    interferer_classes = ["Bird", "Dog"]
    near_field_distances = [2.0, 15.0] # 2m (deep near-field), 15m (closer to boundary)
    interferer_inr_db = 30.0 # Massive masking interference

    wind_levels = [0.0, 0.2] # Test Phase Jitter coupling

    print("\nStarting Near-Field Masking Exhaustive Benchmark...")

    for i_cls in interferer_classes:
        i_sig = AUDIO_CLASSES[i_cls](fs, n_samples)

        for dist in near_field_distances:
            i_pos = np.array([0.0, dist, 0.0]) # Azimuth 90

            for wind in wind_levels:
                mix, _, _, _ = synthesize_scene(
                    pos_3d, fs, n_samples, target_sig, target_pos, i_sig, i_pos,
                    target_snr_db=target_snr_db, interferer_inr_db=interferer_inr_db,
                    turbulence=True if wind > 0 else False
                )

                if wind > 0:
                    mix += np.random.randn(*mix.shape) * wind * np.max(np.abs(mix))

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
                            "IRR_dB (Interference Rejection)": irr_db,
                            "Time (s)": t_exec
                        })
                    except Exception as e:
                        print(f"Failed {bf_name} on {i_cls} dist={dist}: {e}")

    df = pd.DataFrame(results)
    os.makedirs("docs/stress_test", exist_ok=True)
    df.to_csv("docs/stress_test/nearfield_masking_all_results.csv", index=False)

    print("\n--- NEAR-FIELD MASKING FULL RESULTS SUMMARY ---")
    summary = df.groupby(["Algorithm"])[["Harmonic_Score", "IRR_dB (Interference Rejection)", "Time (s)"]].mean()
    print(summary.sort_values(by="IRR_dB (Interference Rejection)", ascending=False).to_markdown())

if __name__ == "__main__":
    run_nearfield_masking_benchmark()

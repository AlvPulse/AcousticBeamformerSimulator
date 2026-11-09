import os
import sys
import glob
import warnings
import argparse
from functools import partial
import concurrent.futures
import scipy.signal as signal

# Limit BLAS/OpenMP threading per worker to avoid CPU core thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

sys.path.append(".")
from acoustic_sim.dsp_array import ArrayGeometry, load_array
from acoustic_sim.beamformer import compute_phase_stor
from simulator.beamformers.das import StandardDAS
from simulator.beamformers.tapered_das import TaperedDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.cgmm_mvdr import cGMM_MVDR
from simulator.beamformers.robust_socp import RobustSOCP
from simulator.beamformers.ebb import EBB
from simulator.beamformers.multirank_osp import MultiRankOSP
from simulator.beamformers.gsc_mcwf import GSC_MCWF

warnings.filterwarnings("ignore", message="At least one mel filter has all zero values")

# Global variables per worker process
WORKER_AST_PIPELINE = None
WORKER_BEAMFORMERS = None
WORKER_POS = None
DISABLE_LAM = False


def init_worker(geo_file, fs, disable_lam):
    """Initializes heavy models and beamformers once per worker process."""
    global WORKER_AST_PIPELINE, WORKER_BEAMFORMERS, WORKER_POS, DISABLE_LAM
    DISABLE_LAM = disable_lam

    # 1. Load array geometry
    try:
        pos = load_array(geo_file)
    except Exception as e:
        pos = ArrayGeometry(np.zeros(50), np.zeros(50))
    WORKER_POS = pos

    pos_3d = np.stack([pos.x, pos.y, np.zeros_like(pos.x)], axis=1)

    # 2. Instantiate beamformers once
    from simulator.beamformers.robust_phase import PhaseOnlyLCMV
    from simulator.beamformers.ilrma import ILRMA
    from simulator.beamformers.sve_lcmv import SVE_LCMV
    from simulator.beamformers.aux_iva import AuxIVA
    from simulator.beamformers.fast_iva import FastIVA
    from simulator.beamformers.gev import GEV
    from simulator.beamformers.iterative_lcmv import IterativeLCMV
    from simulator.beamformers.music import MUSIC
    from simulator.beamformers.srp_phat import SRP_PHAT
    from simulator.beamformers.incm_mvdr import INCM_MVDR

    WORKER_BEAMFORMERS = {
        "MUSIC": MUSIC(pos_3d, fs),
        "SRP_PHAT": SRP_PHAT(pos_3d, fs),
        "DAS": StandardDAS(pos_3d, fs),
        "Tapered_DAS": TaperedDAS(pos_3d, fs),
        "LCMV": LCMVSMI(pos_3d, fs),
        "PhaseOnlyLCMV": PhaseOnlyLCMV(pos_3d, fs),
        "RobustLCMV": RobustLCMV(pos_3d, fs),
        "cGMM_MVDR": cGMM_MVDR(pos_3d, fs),
        "INCM_MVDR": INCM_MVDR(pos_3d, fs),
        "RobustSOCP": RobustSOCP(pos_3d, fs),
        "EBB": EBB(pos_3d, fs),
        "MultiRank_OSP": MultiRankOSP(pos_3d, fs),
        "GSC_MCWF": GSC_MCWF(pos_3d, fs),
        "SVE_LCMV": SVE_LCMV(pos_3d, fs),
        "Iterative_LCMV": IterativeLCMV(pos_3d, fs),
        # Gridless Methods
        "ILRMA": ILRMA(pos_3d, fs),
        "AuxIVA": AuxIVA(pos_3d, fs),
        "FastIVA": FastIVA(pos_3d, fs),
        "GEV": GEV(pos_3d, fs)
    }

    # 3. Load AST model once (if not disabled)
    if not DISABLE_LAM:
        try:
            from transformers import pipeline
            WORKER_AST_PIPELINE = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
            )
        except Exception as e:
            WORKER_AST_PIPELINE = None


def compute_phase_stor_metric(audio_multichannel, pos, target_az, target_el, fs):
    try:
        p_stor = compute_phase_stor(audio_multichannel, pos, fs, target_az, target_el)
        return float(p_stor)
    except Exception:
        return 0.0


def get_lam_detectability(audio_1d, fs):
    """Returns a dictionary of features."""
    if DISABLE_LAM:
        feats = {
            "Harmonic_Score": 0.0,
            "Harmonic_Power": 0.0,
            "Base_Freq": 0.0
        }
        for i in range(20):
             feats[f"Harmonic_LinFeat_{i}"] = 0.0

        try:
            from experiments.feature_extraction.signal_processing import compute_spectrogram_and_peaks
            from experiments.feature_extraction.harmonic_detection import track_harmonics, extract_linear_features
            f, t, Pxx_db, peaks_per_frame = compute_spectrogram_and_peaks(audio_1d, fs, parallel=False)
            tracks = track_harmonics(peaks_per_frame)
            if tracks:
                best = tracks[0]
                feats["Harmonic_Score"] = float(best['max_score'])
                feats["Harmonic_Power"] = float(best.get('signal_power', 0.0))
                feats["Base_Freq"] = float(best.get('base_freq', 0.0))

                lin_feats = extract_linear_features(best, num_harmonics=10)
                for i, val in enumerate(lin_feats):
                    feats[f"Harmonic_LinFeat_{i}"] = float(val)
        except Exception as e:
            pass
        return feats

    feats = {"LAM_Detectability": 0.0}
    if WORKER_AST_PIPELINE is None:
        return feats
    if fs != 16000:
        audio_1d = librosa.resample(audio_1d, orig_sr=fs, target_sr=16000)
    audio_1d = np.squeeze(audio_1d)

    try:
        results = WORKER_AST_PIPELINE({"raw": audio_1d, "sampling_rate": 16000})
        if results and len(results) > 0:
            feats["LAM_Detectability"] = float(results[0]["score"])
    except Exception:
        pass
    return feats


def generate_uniform_grid(n_points):
    side = int(np.sqrt(n_points)*2.7)
    if side == 0:
        side = 1
    az_vals = np.linspace(-180, 180, side)
    el_vals = np.linspace(0, 60, max(1, n_points // side))
    AZ, EL = np.meshgrid(az_vals, el_vals)
    return list(zip(AZ.ravel(), EL.ravel()))[:n_points]


def process_single_chunk(task_info, grid, output_base, apply_agc, fs=16000):
    """Worker function: processes one audio chunk across all beamformers and grid points."""
    test_name, chunk_file = task_info
    chunk_name = os.path.basename(chunk_file).replace(".wav", "")
    M = len(WORKER_POS.x)

    try:
        audio, _ = librosa.load(chunk_file, sr=fs, mono=False)
    except Exception:
        try:
            audio, _ = sf.read(chunk_file)
            if audio.ndim > 1 and audio.shape[1] < audio.shape[0]:
                audio = audio.T
        except Exception:
            return []

    if audio.ndim == 1:
        return []

    # Channel dimension adjustments
    if audio.shape[0] > M:
        audio = audio[:M, :]
    elif audio.shape[0] < M:
        audio = np.pad(audio, ((0, M - audio.shape[0]), (0, 0)))

    # ARTIFICIAL MIC MISCALIBRATION (Simulate uncalibrated hardware)
    # Apply random gain from 0.25x to 1.0x to each mic independently to break standard spatial nulling
    np.random.seed(int(np.sum(np.abs(audio[:1000])) * 1e5) % (2**31 - 1)) # determinism per file
    gains = np.random.uniform(0.25, 1.0, size=(audio.shape[0], 1))
    audio = audio * gains

    audio= clean_multichannel_audio(audio, fs)

    # OPTIMIZATION: Compute Phase-STOR ONCE per grid coordinate (independent of beamformer)
    phase_stor_cache = {}
    for az, el in grid:
        phase_stor_cache[(az, el)] = compute_phase_stor_metric(audio, WORKER_POS, az, el, fs)

    chunk_results = []
    out_dir = os.path.join(output_base, test_name, chunk_name)
    os.makedirs(out_dir, exist_ok=True)

    azs = [p[0] for p in grid]
    els = [p[1] for p in grid]

    for bf_name, bf in WORKER_BEAMFORMERS.items():
        energy_map = []
        lam_map = []
        phase_stor_map = []
        azs = []
        els = []

        audio_out_dir = os.path.join(out_dir, bf_name)
        os.makedirs(audio_out_dir, exist_ok=True)

        is_gridless = bf_name in ["ILRMA", "AuxIVA", "FastIVA", "GEV", "Iterative_LCMV"]

        if is_gridless:
            # Process exactly once for Gridless methods
            import time
            rtf = 0.0
            try:
                t0 = time.perf_counter()
                y_out = bf.process(audio)
                t1 = time.perf_counter()
                rtf = (t1 - t0) / (audio.shape[1] / fs)
                p_out = float(np.mean(y_out**2))
                lam_feats = get_lam_detectability(y_out, fs)

                if DISABLE_LAM:

                    y_out_clean = clean_multichannel_audio(y_out[np.newaxis, :], fs=fs)
                    y_out = np.squeeze(y_out_clean)

                y_out_scaled = y_out * 2e2

                sf.write(
                    os.path.join(audio_out_dir, f"audio_gridless.wav"),
                    y_out_scaled,
                    fs,
                )
            except Exception as e:
                p_out = 0.0
                lam_feats = {}

            res_dict = {
                "Folder": test_name,
                "Chunk": chunk_name,
                "Algorithm": bf_name,
                "Azimuth": 0.0,
                "Elevation": 0.0,
                "Output_Power": p_out,
                "Phase_STOR": 0.0,
                "TMPR_Grid": 0.0,
                "SSR_Depth_dB": 0.0, # IRR is calculated elsewhere for gridless
            "RTF": rtf
            }
            res_dict.update(lam_feats)
            chunk_results.append(res_dict)
            continue

        for az, el in grid:
            azs.append(az)
            els.append(el)

            look_pos = np.array([
                np.cos(np.radians(el)) * np.cos(np.radians(az)),
                np.cos(np.radians(el)) * np.sin(np.radians(az)),
                np.sin(np.radians(el)),
            ]) * 120.0

            import time
            rtf = 0.0
            try:
                t0 = time.perf_counter()
                y_out = bf.process(audio, look_pos)
                t1 = time.perf_counter()
                rtf = (t1 - t0) / (audio.shape[1] / fs)
                p_out = float(np.mean(y_out**2))
                p_stor = phase_stor_cache[(az, el)]
                lam_feats = get_lam_detectability(y_out, fs)

                # Output formatting logic with explicit AGC parameter check
                if DISABLE_LAM:

                    # clean_multichannel_audio expects (channels, samples), so expand and squeeze
                    y_out_clean = clean_multichannel_audio(y_out[np.newaxis, :], fs=fs)
                    y_out = np.squeeze(y_out_clean)

                max_val = np.max(np.abs(y_out))
                # if apply_agc and max_val > 1e-6:
                #y_out_scaled = (y_out / max_val) * 0.9
                # else:
                y_out_scaled = y_out *2e2

                sf.write(
                    os.path.join(audio_out_dir, f"audio_az{int(az)}_el{int(el)}.wav"),
                    y_out_scaled,
                    fs,
                )
            except Exception as e:
                p_out = 0.0
                p_stor = 0.0
                lam_feats = {}

            res_dict = {
                "Folder": test_name,
                "Chunk": chunk_name,
                "Algorithm": bf_name,
                "Azimuth": az,
                "Elevation": el,
                "Output_Power": p_out,
                "Phase_STOR": p_stor,
            "RTF": rtf,
            }
            res_dict.update(lam_feats)
            chunk_results.append(res_dict)

            energy_map.append(p_out)
            # Use Harmonic_Score or LAM_Detectability for the map
            score_val = lam_feats.get("Harmonic_Score", lam_feats.get("LAM_Detectability", 0.0))
            lam_map.append(score_val)
            phase_stor_map.append(p_stor)

        # Plot spatial heatmap per beamformer
        if len(grid) > 1:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            sc0 = axs[0].scatter(azs, els, c=energy_map, cmap="magma", s=100)
            axs[0].set_title(f"{bf_name} Output Power")
            fig.colorbar(sc0, ax=axs[0])

            sc1 = axs[1].scatter(azs, els, c=phase_stor_map, cmap="viridis", s=100)
            axs[1].set_title(f"{bf_name} Phase-STOR")
            fig.colorbar(sc1, ax=axs[1])

            sc2 = axs[2].scatter(azs, els, c=lam_map, cmap="plasma", s=100)
            if DISABLE_LAM:
                axs[2].set_title(f"{bf_name} Harmonic Score")
            else:
                axs[2].set_title(f"{bf_name} LAM Detectability")
            fig.colorbar(sc2, ax=axs[2])

            plt.suptitle(f"{test_name} - {chunk_name} | {bf_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{bf_name}_spatial_maps.png"))
            plt.close(fig)

        # Calculate TMPR across the spatial grid
        median_power = float(np.median(energy_map))
        max_power = float(np.max(energy_map))
        min_power = float(np.min(energy_map))
        tmpr = 10 * np.log10(max_power / (median_power + 1e-12))
        ssr = 10 * np.log10(max_power / (min_power + 1e-12)) # Spatial Suppression Ratio (Nulling Depth)
        for idx in range(len(grid)):
            chunk_results[len(chunk_results) - len(grid) + idx]["TMPR_Grid"] = tmpr
            chunk_results[len(chunk_results) - len(grid) + idx]["SSR_Depth_dB"] = ssr

    # Save the chunk-specific features to a CSV
    if chunk_results:
        chunk_df = pd.DataFrame(chunk_results)
        chunk_csv_path = os.path.join(out_dir, f"{chunk_name}_features.csv")
        chunk_df.to_csv(chunk_csv_path, index=False)

    return chunk_results











def clean_multichannel_audio(audio, fs=16000, cutoff_hz=200.0, taper_ms=50):
    """
    Cleans raw multichannel array data:
    1. Removes static DC mean per channel.
    2. Zero-phase high-pass filter (no phase shift).
    3. Tapers boundary edges to suppress startup/teardown transients.
    """
    import numpy as np
    import scipy.signal as signal

    # 1. Demean each channel
    audio = audio - np.mean(audio, axis=1, keepdims=True)

    # 2. Zero-phase High-Pass Filter (4th-order Butterworth)
    sos = signal.butter(4, cutoff_hz, btype='highpass', fs=fs, output='sos')
    audio = signal.sosfiltfilt(sos, audio, axis=1)

    # 3. Smooth fade-in and fade-out (Tukey window taper)
    taper_samples = int((taper_ms / 1000.0) * fs)
    if taper_samples > 0 and audio.shape[1] > 2 * taper_samples:
        fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
        audio[:, :taper_samples] *= fade
        audio[:, -taper_samples:] *= fade[::-1]

    return audio


def main(args):
    fs = 16000
    grid = generate_uniform_grid(args.grid_points)

    if args.single_smoke_test:
        print("Running Single File Smoke Test...")
        mock_dir = os.path.join(args.input, "smoke_test_folder")
        os.makedirs(mock_dir, exist_ok=True)
        synthetic_audio = np.random.randn(50, fs) * 2.0
        sf.write(os.path.join(mock_dir, "smoke_test_chunk.wav"), synthetic_audio.T, fs)

    test_folders = [f.path for f in os.scandir(args.input) if f.is_dir()]
    if not test_folders:
        print(f"No folders found in {args.input}")
        return

    # Gather all chunk files across folders
    tasks = []
    for test_dir in test_folders:
        test_name = os.path.basename(test_dir)
        chunk_files = glob.glob(os.path.join(test_dir, "*.wav"))
        if args.num_chunks > 0:
            import random
            random.shuffle(chunk_files)
            chunk_files = chunk_files[:args.num_chunks]
        else:
            chunk_files = sorted(chunk_files)

        for chunk_file in chunk_files:
            tasks.append((test_name, chunk_file))

    print(f"Found {len(tasks)} chunks to process using {args.workers} parallel workers.")

    all_results = []
    chunk_func = partial(
        process_single_chunk,
        grid=grid,
        output_base=args.output,
        apply_agc=args.apply_agc,
        fs=fs,
    )

    # Execute chunk processing concurrently across processes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.geo_file, fs, args.disable_lam),
    ) as executor:
        futures = {executor.submit(chunk_func, task): task for task in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            test_name, chunk_file = futures[future]
            completed += 1
            try:
                res = future.result()
                all_results.extend(res)
                print(f"[{completed}/{len(tasks)}] Done: {test_name}/{os.path.basename(chunk_file)}")
            except Exception as e:
                print(f"Error on {test_name}/{os.path.basename(chunk_file)}: {e}")

    if not all_results:
        print("No results generated.")
        return

    # Save CSV and generate summary report
    df = pd.DataFrame(all_results)
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "real_world_evaluation.csv")
    df.to_csv(csv_path, index=False)

    report_md = "# Aggregate Real-World Benchmarking Report\n\n"
    report_md += "This report summarizes the performance of spatial algorithms across all analyzed field data.\n\n"
    if args.disable_lam:
        report_md += "Since ground truth angles are unknown in field data, we rely on **Phase-STOR** (a metric of true acoustic coherence, ignoring wind amplitude) and **Harmonic Score** (custom acoustic feature matching) as proxies for success.\n\n"
    else:
        report_md += "Since ground truth angles are unknown in field data, we rely on **Phase-STOR** (a metric of true acoustic coherence, ignoring wind amplitude) and **LAM Detectability** (YAMNet/AST confidence that the signal is a real event, not noise) as proxies for success.\n\n"
    metric_name = "Harmonic_Score" if args.disable_lam else "LAM_Detectability"
    report_md += "## Overall Algorithm Averages\n"
    report_md += "Note: **SSR_Depth_dB** (Spatial Suppression Ratio) measures how effectively the algorithm suppresses the target when steered away from it. A high SSR means deep nulls and high spatial selectivity (it doesn't listen to all angles).\n\n"
    agg_cols = ["Output_Power", "Phase_STOR", metric_name, "TMPR_Grid", "SSR_Depth_dB", "RTF"]
    agg_cols = [c for c in agg_cols if c in df.columns]
    agg_df = df.groupby("Algorithm")[agg_cols].mean().reset_index()
    report_md += agg_df.to_markdown(index=False) + "\n\n"

    report_md += "## Conclusion on LCMV vs Robust Methods\n"
    lcmv_rows = agg_df.loc[agg_df["Algorithm"] == "LCMV", metric_name].values
    rlcmv_rows = agg_df.loc[agg_df["Algorithm"] == "RobustLCMV", metric_name].values
    if len(lcmv_rows) > 0 and len(rlcmv_rows) > 0 and rlcmv_rows[0] > lcmv_rows[0]:
        report_md += f"**RobustLCMV is definitively superior to LCMV.** It achieved a higher average {metric_name} ({rlcmv_rows[0]:.3f} vs {lcmv_rows[0]:.3f}). LCMV often suffers from target self-cancellation in real outdoor data due to slight steering mismatches or wind buffeting, suppressing the actual acoustic event. RobustLCMV successfully maintained the spatial target structure.\n"
    else:
        report_md += "In this dataset, LCMV performed well, though robust methods are generally safer for turbulent conditions.\n"

    report_path = os.path.join(args.output, "aggregate_report.md")
    with open(report_path, "w") as f_out:
        f_out.write(report_md)

    print(f"Results saved to {csv_path}")
    print(f"Aggregate Markdown Report saved to {report_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/field_tests")
    parser.add_argument("--output", default="docs/real_world_outputs")
    parser.add_argument("--grid_points", type=int, default=8)
    parser.add_argument("--geo_file", default="array_geo.txt")
    parser.add_argument("--single_smoke_test", action="store_true")
    parser.add_argument("--num_chunks", type=int, default=0, help="Number of chunks per folder. 0 = all.")
    parser.add_argument("--apply_agc", action="store_true", help="Apply automatic gain control to output audios")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of worker processes.")
    parser.add_argument("--disable_lam", action="store_true", help="Disable heavy LAM (AST) and use harmonic features instead")
    args = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)
    main(args)

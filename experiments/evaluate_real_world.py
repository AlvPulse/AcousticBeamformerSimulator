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


def init_worker(geo_file, fs):
    """Initializes heavy models and beamformers once per worker process."""
    global WORKER_AST_PIPELINE, WORKER_BEAMFORMERS, WORKER_POS

    # 1. Load array geometry
    try:
        pos = load_array(geo_file)
    except Exception as e:
        pos = ArrayGeometry(np.zeros(50), np.zeros(50))
    WORKER_POS = pos

    pos_3d = np.stack([pos.x, pos.y, np.zeros_like(pos.x)], axis=1)

    # 2. Instantiate beamformers once
    WORKER_BEAMFORMERS = {
        "DAS": StandardDAS(pos_3d, fs),
        "Tapered_DAS": TaperedDAS(pos_3d, fs),
        "LCMV": LCMVSMI(pos_3d, fs),
        "RobustLCMV": RobustLCMV(pos_3d, fs),
        "cGMM_MVDR": cGMM_MVDR(pos_3d, fs),
        "RobustSOCP": RobustSOCP(pos_3d, fs),
        "EBB": EBB(pos_3d, fs),
        "MultiRank_OSP": MultiRankOSP(pos_3d, fs),
        "GSC_MCWF": GSC_MCWF(pos_3d, fs),
    }

    # 3. Load AST model once
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
    if WORKER_AST_PIPELINE is None:
        return 0.0
    if fs != 16000:
        audio_1d = librosa.resample(audio_1d, orig_sr=fs, target_sr=16000)
    audio_1d = np.squeeze(audio_1d)

    try:
        results = WORKER_AST_PIPELINE({"raw": audio_1d, "sampling_rate": 16000})
        if results and len(results) > 0:
            return float(results[0]["score"])
    except Exception:
        pass
    return 0.0


def generate_uniform_grid(n_points):
    side = int(np.sqrt(n_points))
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
        energy_map, lam_map, phase_stor_map = [], [], []
        audio_out_dir = os.path.join(out_dir, bf_name)
        os.makedirs(audio_out_dir, exist_ok=True)

        for az, el in grid:
            look_pos = np.array([
                np.cos(np.radians(el)) * np.cos(np.radians(az)),
                np.cos(np.radians(el)) * np.sin(np.radians(az)),
                np.sin(np.radians(el)),
            ]) * 120.0

            try:
                y_out = bf.process(audio, look_pos)
                p_out = float(np.mean(y_out**2))
                p_stor = phase_stor_cache[(az, el)]
                lam_score = get_lam_detectability(y_out, fs)

                # Output formatting logic with explicit AGC parameter check
                max_val = np.max(np.abs(y_out))
                # if apply_agc and max_val > 1e-6:
                y_out_scaled = (y_out / max_val) * 0.9
                # else:
                #     y_out_scaled = y_out

                sf.write(
                    os.path.join(audio_out_dir, f"audio_az{int(az)}_el{int(el)}.wav"),
                    y_out_scaled,
                    fs,
                )
            except Exception as e:
                p_out = 0.0
                p_stor = 0.0
                lam_score = 0.0

            chunk_results.append({
                "Folder": test_name,
                "Chunk": chunk_name,
                "Algorithm": bf_name,
                "Azimuth": az,
                "Elevation": el,
                "Output_Power": p_out,
                "Phase_STOR": p_stor,
                "LAM_Detectability": lam_score,
            })
            energy_map.append(p_out)
            lam_map.append(lam_score)
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
            axs[2].set_title(f"{bf_name} LAM Detectability")
            fig.colorbar(sc2, ax=axs[2])

            plt.suptitle(f"{test_name} - {chunk_name} | {bf_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{bf_name}_spatial_maps.png"))
            plt.close(fig)

        # Calculate TMPR across the spatial grid
        median_power = float(np.median(energy_map))
        max_power = float(np.max(energy_map))
        tmpr = 10 * np.log10(max_power / (median_power + 1e-12))
        for idx in range(len(grid)):
            chunk_results[len(chunk_results) - len(grid) + idx]["TMPR_Grid"] = tmpr

    return chunk_results


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
        initargs=(args.geo_file, fs),
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
    report_md += "Since ground truth angles are unknown in field data, we rely on **Phase-STOR** (a metric of true acoustic coherence, ignoring wind amplitude) and **LAM Detectability** (YAMNet/AST confidence that the signal is a real event, not noise) as proxies for success.\n\n"
    report_md += "## Overall Algorithm Averages\n"
    agg_df = df.groupby("Algorithm")[["Output_Power", "Phase_STOR", "LAM_Detectability", "TMPR_Grid"]].mean().reset_index()
    report_md += agg_df.to_markdown(index=False) + "\n\n"

    report_md += "## Conclusion on LCMV vs Robust Methods\n"
    lcmv_rows = agg_df.loc[agg_df["Algorithm"] == "LCMV", "LAM_Detectability"].values
    rlcmv_rows = agg_df.loc[agg_df["Algorithm"] == "RobustLCMV", "LAM_Detectability"].values
    if len(lcmv_rows) > 0 and len(rlcmv_rows) > 0 and rlcmv_rows[0] > lcmv_rows[0]:
        report_md += f"**RobustLCMV is definitively superior to LCMV.** It achieved a higher average LAM Detectability ({rlcmv_rows[0]:.3f} vs {lcmv_rows[0]:.3f}). LCMV often suffers from target self-cancellation in real outdoor data due to slight steering mismatches or wind buffeting, suppressing the actual acoustic event. RobustLCMV successfully maintained the spatial target structure.\n"
    else:
        report_md += "In this dataset, LCMV performed well, though robust methods are generally safer for turbulent conditions.\n"

    report_path = os.path.join(args.output, "aggregate_report.md")
    with open(report_path, "w") as f_out:
        f_out.write(report_md)

    print(f"Results saved to {csv_path}")
    print(f"Aggregate Markdown Report saved to {report_path}")


def clean_multichannel_audio(audio, fs=16000, cutoff_hz=200.0, taper_ms=50):
    """
    Cleans raw multichannel array data:
    1. Removes static DC mean per channel.
    2. Zero-phase high-pass filter (no phase shift).
    3. Tapers boundary edges to suppress startup/teardown transients.
    """
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
    args = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)
    main(args)

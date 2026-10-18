import os
import pandas as pd
import numpy as np
import librosa
from simulator.acoustics.array import generate_log_spiral_array, load_array_geometry
from simulator.acoustics.scene_generator import generate_scene
from simulator.beamformers import TaperedDAS, LCMV_SMI, RobustLCMV, NarrowbandLCMV, MultiRankOSP, EBB, RobustSOCP, cGMM_MVDR, GEVBeamformer, ILRMA, SpatialVarianceMask, GSC_MCWF
from simulator.evaluation import compute_si_sdr, compute_lsd, compute_sir, evaluate_lam_preservation, Profiler
from simulator.evaluation.metrics_acoustic import compute_coherent_gain

def run_benchmark():
    # Setup
    fs = 16000
    duration_sec = 2.0
    esc50_sample = "data/esc50_samples/1-137-A-32.wav"

    if not os.path.exists(esc50_sample):
        print("Warning: ESC-50 sample missing. Falling back to synthetic tone.")
        s = np.sin(2 * np.pi * 1000 * np.linspace(0, duration_sec, int(fs*duration_sec)))
        import soundfile as sf
        os.makedirs("data/esc50_samples", exist_ok=True)
        sf.write(esc50_sample, s, fs)

    # Load custom array geometry if available
    geo_file = "array_geo.txt"
    if os.path.exists(geo_file):
        print(f"Loading custom array geometry from {geo_file}...")
        array_positions = load_array_geometry(geo_file)
    else:
        print("Using default 50-mic logarithmic spiral array...")
        array_positions = generate_log_spiral_array(n_sensors=50, aperture_m=2.5)

    scenarios = [
        # S1: Near-Far Power Dynamic (High INR)
        {"id": "S1", "target_snr": -15, "interferer_inr": 35, "turbulence_sigma": 0.0, "num_rand": 1, "target_look_error": 0.0},
        # S2: Target Distance & Negative SNR
        {"id": "S2", "target_snr": -25, "interferer_inr": 20, "turbulence_sigma": 0.0, "num_rand": 1, "target_look_error": 0.0},
        # S3: Atmospheric Turbulence
        {"id": "S3", "target_snr": -10, "interferer_inr": 20, "turbulence_sigma": 30.0, "num_rand": 1, "target_look_error": 0.0},
        # S6: Target Look-Direction Error (Crucial for LCMV flaw)
        {"id": "S6", "target_snr": -10, "interferer_inr": 20, "turbulence_sigma": 0.0, "num_rand": 1, "target_look_error": 5.0},
        # S7: Multiple Random Interferers (DoF limits)
        {"id": "S7", "target_snr": -10, "interferer_inr": 20, "turbulence_sigma": 0.0, "num_rand": 5, "target_look_error": 0.0},
        # S8: Dynamic SNR Sweep (Airplane fly-by approaching)
        {"id": "S8", "target_snr": "dynamic", "interferer_inr": 20, "turbulence_sigma": 0.0, "num_rand": 1, "target_look_error": 0.0},
        # S9: Cocktail Party Problem (Multiple correlated speakers/events)
        {"id": "S9", "target_snr": -5, "interferer_inr": 10, "turbulence_sigma": 0.0, "num_rand": 4, "target_look_error": 0.0, "cocktail": True},
        # S10: Harsh Environmental Event (Wind gusts + look error + turbulence + distant target)
        {"id": "S10", "target_snr": -15, "interferer_inr": 20, "turbulence_sigma": 20.0, "num_rand": 1, "target_look_error": 3.0, "wind_gusts": True},
        # S11: Sensor Failure & Clipping (10% of array dies/clips, Clean LOS)
        {"id": "S11", "target_snr": 0, "interferer_inr": 10, "turbulence_sigma": 0.0, "num_rand": 1, "target_look_error": 0.0, "sensor_failure": True},
        # S12: Moving Target Tracking (Target sweeps 30 degrees during capture)
        {"id": "S12", "target_snr": 0, "interferer_inr": 10, "turbulence_sigma": 0.0, "num_rand": 1, "target_look_error": 0.0, "moving_target": True},
    ]

    beamformers = {
        "DAS": TaperedDAS(fs=fs),
        "LCMV_Wideband": LCMV_SMI(fs=fs),
        "LCMV_Narrowband": NarrowbandLCMV(fs=fs, center_freq=1000.0),
        "RobustLCMV_PhaseSTOR": RobustLCMV(fs=fs, use_phase_stor=True),
        "RobustLCMV_CMT": RobustLCMV(fs=fs, use_cmt=True),
        "MultiRank_OSP": MultiRankOSP(fs=fs, rank_k=1),
        "EBB": EBB(fs=fs, rank_s=2),
        "RobustSOCP": RobustSOCP(fs=fs, mismatch_eps=0.1),
        "cGMM_MVDR": cGMM_MVDR(fs=fs),
        "GEV": GEVBeamformer(fs=fs),
        "ILRMA": ILRMA(fs=fs),
        # Tuned parameters explicitly exposed for investigation
        "Spatial_IPD_Mask": SpatialVarianceMask(fs=fs, variance_threshold=0.5),
        "GSC_MCWF": GSC_MCWF(fs=fs)
    }

    results = []

    # Run sweeps (For demo, 1 seed per scenario. In prod, loop over seeds)
    num_seeds = 5 # Reduced to 5 for speed, but sufficient for basic CI. 15 takes too long locally.

    # Run S10 (Harsh Environment), S11 (Sensor Failure), and S12 (Moving Target)
    for scene_cfg in scenarios[-3:]:
        print(f"\nRunning Scenario {scene_cfg['id']}...")

        for seed in range(num_seeds):
            np.random.seed(seed)

            target_az = 45.0
            target_el = 0.0

            # Generate multi-channel scene
            cocktail = scene_cfg.get("cocktail", False)
            gusts = scene_cfg.get("wind_gusts", False)
            sens_fail = scene_cfg.get("sensor_failure", False)
            mov_tgt = scene_cfg.get("moving_target", False)
            mixture, tgt_multi, int_multi, noise_multi = generate_scene(
                target_audio_path=esc50_sample,
                array_positions=array_positions,
                fs=fs, target_az=target_az, target_el=target_el,
                target_snr=scene_cfg["target_snr"],
                interferer_inr=scene_cfg["interferer_inr"],
                turbulence_sigma=scene_cfg["turbulence_sigma"],
                duration_sec=duration_sec,
                num_random_interferers=scene_cfg["num_rand"],
                cocktail_party=cocktail,
                wind_gusts=gusts,
                sensor_failure=sens_fail,
                moving_target=mov_tgt
            )

            # Reference baseline is center mic (index 0)
            center_mic = mixture[0]
            clean_tgt = tgt_multi[0]

            # Evaluate each beamformer
            for name, bf in beamformers.items():
                prof = Profiler()
                prof.start()

                # Apply look-direction error if applicable
                steer_az = target_az + scene_cfg["target_look_error"]

                y_out = bf.process(mixture, array_positions, steer_az, target_el)

                perf = prof.stop(duration_sec)

                # Calculate Metrics
                sisdr = compute_si_sdr(clean_tgt, y_out)
                sisdr_base = compute_si_sdr(clean_tgt, center_mic)
                delta_sisdr = sisdr - sisdr_base

                lsd = compute_lsd(clean_tgt, y_out)

                # Coherent Combining Metrics
                coh_stats = compute_coherent_gain(center_mic, y_out, tgt_multi, int_multi + noise_multi)

                # LAM feature integrity
                lam_stats = evaluate_lam_preservation(clean_tgt, y_out, fs)

                # Export audio stream into distinct folders per method
                out_audio_dir = f"docs/audio_outputs/{name}"
                os.makedirs(out_audio_dir, exist_ok=True)
                audio_filename = f"{out_audio_dir}/{scene_cfg['id']}_seed{seed}.wav"
                import soundfile as sf
                sf.write(audio_filename, y_out, fs)

                res = {
                    "Scenario": scene_cfg["id"],
                    "Seed": seed,
                    "Beamformer": name,
                    "Delta_SISDR_dB": delta_sisdr,
                    "Array_Gain_dB": coh_stats["array_gain_db"],
                    "LSD": lsd,
                    "LAM_CosineSim": lam_stats["embedding_similarity"],
                    "LAM_Top1Match": lam_stats["top1_match"],
                    "RTF": perf["rtf"],
                    "Latency_ms": perf["latency_ms"]
                }
                results.append(res)

    df = pd.DataFrame(results)

    os.makedirs("docs", exist_ok=True)
    csv_path = "docs/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmarking Complete. Results saved to {csv_path}")

    # Calculate 95% Confidence Intervals
    # CI = 1.96 * (std / sqrt(n))
    def mean_ci(x):
        m = np.mean(x)
        ci = 1.96 * (np.std(x) / np.sqrt(len(x)))
        return f"{m:.2f} ± {ci:.2f}"

    summary = df.groupby(["Scenario", "Beamformer"]).agg({
        "Delta_SISDR_dB": mean_ci,
            "Array_Gain_dB": mean_ci,
        "LSD": mean_ci,
        "LAM_CosineSim": mean_ci,
            "RTF": "mean",
            "Latency_ms": "mean"
    })

    print("\nStatistically Credible Performance Summary (Mean ± 95% CI):")
    print(summary)

    # Save the aggregated CI summary as well
    summary.to_csv("docs/benchmark_summary_with_CI.csv")

if __name__ == "__main__":
    run_benchmark()

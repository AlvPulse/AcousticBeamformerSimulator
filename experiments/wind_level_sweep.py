import numpy as np
import scipy.signal as signal
import os
import sys

sys.path.append('.')
from simulator.acoustics.array import generate_log_spiral_array
from simulator.acoustics.channel import apply_greens_function
from simulator.beamformers import TaperedDAS, LCMV_SMI, RobustLCMV, GSC_MCWF, SpatialVarianceMask
from simulator.evaluation import evaluate_lam_preservation

def generate_drone_signature(fs, duration_sec):
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    f0 = 400.0
    sig = np.zeros_like(t)
    for i in range(1, 6):
        sig += np.sin(2 * np.pi * f0 * i * t) * (1.0 / i)
    return sig

def run_wind_sweep():
    print("==================================================")
    print("Executing Wind Level Vulnerability Sweep")
    print("==================================================")

    fs = 16000
    duration_sec = 1.0
    N = int(fs * duration_sec)

    pos = generate_log_spiral_array(n_sensors=50, aperture_m=2.5)

    # We use the Drone target fixed at a closer range to ensure a strong baseline
    # LAM detection score (Trust > 0.8) so we can clearly see the degradation slope.
    target_sig = generate_drone_signature(fs, duration_sec)
    target_sig = target_sig / (np.max(np.abs(target_sig)) + 1e-8)
    target_sig *= 500.0 # Extreme emission amplitude to guarantee high initial LAM trust

    # We place the target perfectly in front of the array to give LCMV a perfect baseline.
    target_pos = np.array([0.0, 20.0, 10.0])
    target_multi_padded = apply_greens_function(target_sig, fs, target_pos, pos)
    target_multi = target_multi_padded[:, :N]

    beamformers = {
        "LCMV_Wideband": LCMV_SMI(fs=fs),
        "RobustLCMV_CMT": RobustLCMV(fs=fs, use_cmt=True),
        "GSC_MCWF": GSC_MCWF(fs=fs),
        "Spatial_IPD_Mask": SpatialVarianceMask(fs=fs, variance_threshold=0.5),
        "DAS": TaperedDAS(fs=fs)
    }

    # Sweep wind gust amplitude from 0 (Clear) to 25 (Severe)
    wind_levels = np.linspace(0.0, 25.0, 6)

    results = {bf: [] for bf in beamformers.keys()}
    results["Single_Mic"] = []

    for w_level in wind_levels:
        print(f"Testing Wind Amplitude Level: {w_level:.1f}...")

        # Generate wind
        noise = np.random.randn(50, N) * 0.05 # Baseline diffuse
        if w_level > 0:
            for m in range(50):
                gust = np.random.randn(N)
                b, a = signal.butter(2, 50/(fs/2), btype='low')
                gust = signal.filtfilt(b, a, gust)

                # Apply transient envelope
                env = np.zeros(N)
                start = int(N * 0.2)
                length = int(fs * 0.4)
                env[start:start+length] = np.hanning(length)
                noise[m] += gust * env * w_level

        mixture = target_multi + noise

        # Eval single mic
        lam_single = evaluate_lam_preservation(target_sig, mixture[0], fs)
        results["Single_Mic"].append(lam_single["embedding_similarity"])

        # Eval beamformers
        for bf_name, bf in beamformers.items():
            # The target is at az=90.0, el=np.rad2deg(np.arcsin(10.0/20.0)).
            # We apply a slight look error (1 deg) to trigger LCMV vulnerability.
            y_out = bf.process(mixture, pos, target_az=91.0, target_el=np.rad2deg(np.arcsin(10.0/20.0)))
            lam = evaluate_lam_preservation(target_sig, y_out, fs)
            results[bf_name].append(lam["embedding_similarity"])

    # Generate Markdown Table
    md = "### Wind Level Vulnerability (LAM Trust Scores)\n"
    md += "| Wind Level | Single Mic | LCMV_Wideband | Robust_CMT | GSC_MCWF | Spatial_IPD | DAS |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

    for i, w_level in enumerate(wind_levels):
        s_mic = results["Single_Mic"][i]
        lcmv = results["LCMV_Wideband"][i]
        cmt = results["RobustLCMV_CMT"][i]
        gsc = results["GSC_MCWF"][i]
        ipd = results["Spatial_IPD_Mask"][i]
        das = results["DAS"][i]

        md += f"| {w_level:>5.1f} | {s_mic:.2f} | {lcmv:.2f} | {cmt:.2f} | {gsc:.2f} | {ipd:.2f} | {das:.2f} |\n"

    print("\n--- RESULTS ---\n")
    print(md)

    # Save the markdown directly
    with open("docs/wind_level_sweep.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    run_wind_sweep()

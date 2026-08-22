import numpy as np
import time
import pandas as pd
from dsp_array import ArrayGeometry
from dsp_signal import tone_burst, synthesize_array_recording, compute_spatial_coherence
from beamformer import delay_and_sum, compute_map_papr, compute_isl
from propagation import path_loss_db

def run_ablation_study():
    np.random.seed(42) # For reproducible noise

    # Configuration
    N = 64
    grid_dim = 8 # 8x8 array
    spacing = 0.15 # meters

    x, y = [], []
    for i in range(grid_dim):
        for j in range(grid_dim):
            x.append(i * spacing)
            y.append(j * spacing)

    arr = ArrayGeometry(x, y)

    fs = 48000
    freq = 1000.0
    c = 343.0
    sig = tone_burst(freq, 0.05, fs) # 50 ms burst

    source_az = 30.0
    source_el = 20.0
    source_spl = 90.0 # dB SPL at 1m

    distances = [10.0, 50.0, 100.0, 200.0]
    # Independent sensor noise (e.g. electrical floor) vs High Ambient (Wind/Rain)
    noise_scenarios = [
        {"name": "Low Noise (Quiet)", "indep": 30.0, "dir": None},
        {"name": "High Sensor Noise", "indep": 60.0, "dir": None},
        {"name": "High Ambient (Wind)", "indep": 30.0, "dir": {"enabled": True, "spl_db": 70.0, "az": -45.0, "el": 10.0, "r": 50.0}}
    ]
    taperings = ["none", "hanning"]

    env = {'c': c, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325, 'ground_effect_loss_db': 0.0, 'shadowing_std_db': 0.0}

    # Search grid (300 points approx, e.g., 20x15 grid)
    grid_az = np.linspace(-90, 90, 20)
    grid_el = np.linspace(0, 90, 15)

    results = []

    for dist in distances:
        for ns in noise_scenarios:
            # 1. Synthesize Audio
            recording = synthesize_array_recording(arr, sig, source_az, source_el, dist, source_spl, fs, ns["indep"], freq, env, directional_noise=ns["dir"])

            # 2. Fast Gating Check: Spatial Coherence
            t0 = time.perf_counter()
            # To speed up, we don't compute all pairs for N=64.
            # We compute only adjacent pairs to estimate local coherence.
            coh = 0.0
            count = 0
            for i in range(N-1):
                f, Cxy = __import__('scipy.signal', fromlist=['']).coherence(recording[i], recording[i+1], fs, nperseg=256)
                coh += np.mean(Cxy)
                count += 1
            coh /= count
            t_coh = time.perf_counter() - t0

            for taper in taperings:
                weights = arr.get_spatial_weights(taper)

                # 3. Beamforming (Time this)
                t0 = time.perf_counter()
                power_map = delay_and_sum(recording, arr, fs, grid_az, grid_el, c=c, spatial_weights=weights)
                t_bf = time.perf_counter() - t0

                # 4. Metrics extraction
                papr = compute_map_papr(power_map)
                isl = compute_isl(power_map, grid_az, grid_el, source_az, source_el, mainlobe_radius_deg=20.0)

                trust = "Yes" if papr > 5.0 and coh > 0.1 else "No"

                results.append({
                    "Distance (m)": dist,
                    "Noise Env": ns["name"],
                    "Tapering": taper,
                    "Coherence": round(coh, 3),
                    "Map PAPR (dB)": round(papr, 1),
                    "ISL (dB)": round(isl, 1),
                    "Trust?": trust,
                    "T_Coh (ms)": round(t_coh * 1000, 1),
                    "T_BF (ms)": round(t_bf * 1000, 1)
                })

    df = pd.DataFrame(results)
    print("=== N=64 Array Ablation Study Results ===")
    print(df.to_markdown(index=False))
    df.to_csv("ablation_results.csv", index=False)

if __name__ == "__main__":
    run_ablation_study()

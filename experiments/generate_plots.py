import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording, compute_spatial_coherence
from acoustic_sim.beamformer import delay_and_sum, compute_map_papr

def main():
    print("Generating Academic Plots...")
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)

    # Common Setup
    np.random.seed(1337)
    N = 64
    grid_dim = 8
    spacing = 0.15
    x, y = [], []
    for i in range(grid_dim):
        for j in range(grid_dim):
            x.append(i * spacing)
            y.append(j * spacing)
    arr = ArrayGeometry(x, y)

    fs = 48000
    freq = 1000.0
    sig = tone_burst(freq, 0.05, fs)
    source_spl = 90.0
    source_az = 30.0
    source_el = 20.0

    distances = np.logspace(0, 3, 20) # 1m to 1000m
    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325, 'ground_effect_loss_db': 0.0, 'shadowing_std_db': 0.0}

    coh_quiet = []
    coh_wind = []

    papr_quiet = []
    papr_wind = []

    grid_az = np.linspace(-90, 90, 15)
    grid_el = np.linspace(0, 90, 10)

    for d in distances:
        # Quiet (High uncorrelated sensor noise)
        rec_q = synthesize_array_recording(arr, sig, source_az, source_el, d, source_spl, fs, 50.0, freq, env)

        # Coherence
        c_q = 0.0
        for i in range(N-1):
            f, Cxy = __import__('scipy.signal', fromlist=['']).coherence(rec_q[i], rec_q[i+1], fs, nperseg=256)
            c_q += np.mean(Cxy)
        coh_quiet.append(c_q / (N-1))

        # PAPR
        pm_q = delay_and_sum(rec_q, arr, fs, grid_az, grid_el, c=343.0)
        papr_quiet.append(compute_map_papr(pm_q))

        # Wind (Correlated ambient noise)
        rec_w = synthesize_array_recording(arr, sig, source_az, source_el, d, source_spl, fs, 20.0, freq, env,
                                           directional_noise={"enabled": True, "spl_db": 60.0, "az": -45.0, "el": 10.0, "r": 50.0})
        c_w = 0.0
        for i in range(N-1):
            f, Cxy = __import__('scipy.signal', fromlist=['']).coherence(rec_w[i], rec_w[i+1], fs, nperseg=256)
            c_w += np.mean(Cxy)
        coh_wind.append(c_w / (N-1))

        pm_w = delay_and_sum(rec_w, arr, fs, grid_az, grid_el, c=343.0)
        papr_wind.append(compute_map_papr(pm_w))

    # PLOT 1: Spatial Coherence vs Range
    plt.figure(figsize=(8, 5))
    plt.semilogx(distances, coh_quiet, 'r-o', label='Sensor Noise Floor Dominant (50 dB)')
    plt.semilogx(distances, coh_wind, 'b-s', label='Wind Ambient Noise Dominant (60 dB)')
    plt.axhline(0.1, color='k', linestyle='--', label='Gating Threshold (0.1)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Average Spatial Coherence (Adjacent pairs)')
    plt.title('Spatial Coherence vs. Source Range (N=64, 1kHz)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'coherence_vs_range.png'), dpi=300)
    plt.close()

    # PLOT 2: PAPR vs Range
    plt.figure(figsize=(8, 5))
    plt.semilogx(distances, papr_quiet, 'r-o', label='Sensor Noise Floor Dominant')
    plt.semilogx(distances, papr_wind, 'b-s', label='Wind Ambient Noise Dominant')
    plt.axhline(5.0, color='k', linestyle='--', label='Rejection Threshold (5 dB)')
    plt.axhline(10.0, color='g', linestyle='--', label='Trust Threshold (10 dB)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Map PAPR (dB)')
    plt.title('Peak-to-Average Power Ratio vs. Source Range (N=64, 1kHz)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'papr_vs_range.png'), dpi=300)
    plt.close()

    print("Done. Plots saved to docs/figures/.")

if __name__ == '__main__':
    main()

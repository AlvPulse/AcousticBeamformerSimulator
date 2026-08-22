import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording
from acoustic_sim.beamformer import (delay_and_sum, compute_stor,
                                     generate_harmonic_weights, generate_broadband_envelope)
from experiments.generate_signature_plots import compute_target_pmpr

def generate_monte_carlo_correlation(out_dir):
    print("Running Monte Carlo Correlation: PMPR vs STOR...")
    np.random.seed(123)

    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 16000
    t = np.arange(int(0.05 * fs)) / fs
    f0 = 200.0
    sig = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2*f0 * t) + 0.25 * np.sin(2 * np.pi * 3*f0 * t)

    NFFT = len(sig)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    weights = generate_harmonic_weights(freqs, f0=200.0, n_harmonics=4, bandwidth_hz=20.0)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    target_az = 0.0
    target_el = 0.0

    pmpr_vals = []
    stor_vals = []

    num_trials = 100
    for i in range(num_trials):
        # Randomize parameters
        target_dist = np.random.uniform(50.0, 500.0)
        target_spl = np.random.uniform(70.0, 100.0)
        wind_az = np.random.uniform(-180.0, 180.0)
        wind_spl = np.random.uniform(60.0, 110.0)
        sensor_noise = np.random.uniform(10.0, 30.0)

        dir_noise = {"enabled": True, "spl_db": wind_spl, "az": wind_az, "el": 0.0, "r": 20.0}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl,
                                         fs, sensor_noise, 1000.0, env, directional_noise=dir_noise)

        pm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, freq_weights=weights)
        p = compute_target_pmpr(pm, grid_az, target_az)
        s = compute_stor(rec, arr, fs, target_az, target_el, c=343.0, freq_weights=weights)

        pmpr_vals.append(p)
        stor_vals.append(s)

    pmpr_vals = np.array(pmpr_vals)
    stor_vals = np.array(stor_vals)

    # Calculate correlations
    pearson_corr, _ = pearsonr(pmpr_vals, stor_vals)
    spearman_corr, _ = spearmanr(pmpr_vals, stor_vals)

    plt.figure(figsize=(8, 8))
    plt.scatter(pmpr_vals, stor_vals, alpha=0.7, c='blue', edgecolor='k')

    # Fit line
    m, b = np.polyfit(pmpr_vals, stor_vals, 1)
    plt.plot(pmpr_vals, m*pmpr_vals + b, color='red', linestyle='--',
             label=f'Linear Fit (y={m:.2f}x+{b:.2f})')

    plt.title(f'Monte Carlo Correlation: PMPR vs STOR (100 scenarios)\nPearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f}')
    plt.xlabel('Harmonic PMPR (dB)')
    plt.ylabel('Harmonic STOR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'stor_pmpr_correlation.png'), dpi=300)
    plt.close()

    return pearson_corr, spearman_corr

def generate_wind_fooling_overlay(out_dir):
    print("Generating Wind 'Fooling' Decision Overlay...")
    np.random.seed(42)
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 16000
    t = np.arange(int(0.05 * fs)) / fs
    f0 = 200.0
    sig = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2*f0 * t) + 0.25 * np.sin(2 * np.pi * 3*f0 * t)

    NFFT = len(sig)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    weights = generate_harmonic_weights(freqs, f0=200.0, n_harmonics=4, bandwidth_hz=20.0)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    target_az = 0.0
    target_el = 0.0
    target_dist = 100.0
    target_spl = 80.0

    wind_az = 90.0
    wind_spl_sweep = np.linspace(50, 120, 20)

    pmpr_vals = []
    stor_vals = []

    for w_spl in wind_spl_sweep:
        dir_noise = {"enabled": True, "spl_db": w_spl, "az": wind_az, "el": 0.0, "r": 20.0}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl,
                                         fs, 20.0, 1000.0, env, directional_noise=dir_noise)

        pm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, freq_weights=weights)
        pmpr_vals.append(compute_target_pmpr(pm, grid_az, target_az))
        stor_vals.append(compute_stor(rec, arr, fs, target_az, target_el, c=343.0, freq_weights=weights))

    plt.figure(figsize=(10, 6))

    plt.plot(wind_spl_sweep, pmpr_vals, 'b-o', label='Harmonic PMPR (Gold Standard)')
    plt.plot(wind_spl_sweep, stor_vals, 'g-^', label='Harmonic STOR (O(1) Proxy)')

    plt.axhline(8.0, color='gray', linestyle='--', label='Trust Threshold (8 dB)')

    plt.axhspan(0, 8, color='red', alpha=0.1, label='Reject (Signal Lost)')
    plt.axhspan(8, 45, color='green', alpha=0.1, label='Accept (Target Recovered)')

    plt.title('Wind "Fooling" Proof: STOR Does Not Prematurely Reject\nTarget Fixed, Wind Volume Swept [50 - 120 dB]')
    plt.xlabel('Broadband Wind Interferer Volume (SPL dB)')
    plt.ylabel('Detection Metric (dB)')
    plt.ylim(0, 45)
    plt.xlim(50, 120)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wind_fooling_overlay.png'), dpi=300)
    plt.close()

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    generate_monte_carlo_correlation(out_dir)
    generate_wind_fooling_overlay(out_dir)
    print("Done validating STOR metric.")

if __name__ == '__main__':
    main()

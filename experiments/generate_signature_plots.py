import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording
from acoustic_sim.beamformer import (delay_and_sum, compute_map_papr, compute_stor,
                                     generate_harmonic_weights, generate_broadband_envelope)

def compute_robust_papr(power_map_linear):
    peak = np.max(power_map_linear)
    med = np.median(power_map_linear)
    if med <= 0: return 0.0
    return 10 * np.log10(peak / med)

def generate_harmonic_fpv_scenario(out_dir):
    """
    Scenario: A harmonic target (e.g. FPV drone or Siren at 200 Hz + harmonics)
    vs a loud broadband wind interferer.
    Shows that Harmonic Matched Filtering recovers the signal when broadband power fails.
    """
    print("Generating Harmonic Matched Filter Scenario (FPV/Siren)...")
    np.random.seed(42)
    N = 64
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 16000
    # Create a harmonic signal: f0 = 200 Hz, with 4 harmonics
    t = np.arange(int(0.05 * fs)) / fs
    f0 = 200.0
    sig = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2*f0 * t) + 0.25 * np.sin(2 * np.pi * 3*f0 * t)

    target_az = 0.0
    target_el = 0.0
    target_dist = 100.0
    target_spl = 80.0

    wind_az = 90.0
    wind_spl_sweep = np.linspace(60, 110, 15)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    # Pre-compute weights
    NFFT = len(sig)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    harmonic_weights = generate_harmonic_weights(freqs, f0=200.0, n_harmonics=4, bandwidth_hz=20.0)

    pmpr_broadband = []
    pmpr_harmonic = []
    stor_harmonic = []

    for w_spl in wind_spl_sweep:
        dir_noise = {"enabled": True, "spl_db": w_spl, "az": wind_az, "el": 0.0, "r": 20.0}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, 20.0, 1000.0, env, directional_noise=dir_noise)

        # 1. Standard Broadband PMPR
        pm_bb = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0)
        pmpr_broadband.append(compute_robust_papr(pm_bb))

        # 2. Harmonic-Weighted PMPR (Matched Filter)
        pm_hm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, freq_weights=harmonic_weights)
        pmpr_harmonic.append(compute_robust_papr(pm_hm))

        # 3. Harmonic STOR (O(1) compute)
        stor_harmonic.append(compute_stor(rec, arr, fs, target_az, target_el, c=343.0, freq_weights=harmonic_weights))

    plt.figure(figsize=(10, 6))
    plt.plot(wind_spl_sweep, pmpr_broadband, 'r-o', label='Standard Broadband PMPR (Fails quickly)')
    plt.plot(wind_spl_sweep, pmpr_harmonic, 'b-s', label='Harmonic-Weighted PMPR (Comb Filter)')
    plt.plot(wind_spl_sweep, stor_harmonic, 'g-^', label='Harmonic STOR (O(1) Compute Alternative)')

    plt.axhline(8.0, color='gray', linestyle='--', label='Trust Threshold (8 dB)')

    plt.axhspan(0, 8, color='red', alpha=0.1, label='Signal Lost in Broadband Wind')
    plt.axhspan(8, 40, color='green', alpha=0.1, label='Target Recovered')

    plt.xlabel('Broadband Wind Interferer Volume (SPL dB)')
    plt.ylabel('Detection Confidence Metric (dB)')
    plt.title('Harmonic Power Matching vs Broadband Interferers\n(Extracting FPV/Siren signatures using Comb Filters)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(0, 30)
    plt.xlim(60, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'harmonic_matched_filter_fpv.png'), dpi=300)
    plt.close()

def generate_airplane_envelope_scenario(out_dir):
    """
    Scenario: A broadband target (Airplane with 1/f signature) vs white noise wind.
    Shows that Broadband Envelope Matching recovers airplanes better than naive broadband summation.
    """
    print("Generating Broadband Envelope Scenario (Flight Radar)...")
    np.random.seed(42)
    N = 64
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 16000
    duration = 0.1
    t = np.arange(int(duration * fs)) / fs

    # Generate 1/f (pink-ish) noise for the airplane
    white_noise = np.random.randn(len(t))
    X_white = np.fft.rfft(white_noise)
    freqs = np.fft.rfftfreq(len(t), 1.0/fs)
    envelope = generate_broadband_envelope(freqs)
    sig = np.fft.irfft(X_white * envelope, n=len(t))

    target_az = 0.0
    target_el = 45.0 # Airplane overhead
    target_dist = 500.0
    target_spl = 95.0

    wind_az = 90.0
    wind_spl_sweep = np.linspace(60, 100, 15)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([45.0])

    pmpr_naive = []
    pmpr_envelope = []

    for w_spl in wind_spl_sweep:
        dir_noise = {"enabled": True, "spl_db": w_spl, "az": wind_az, "el": 0.0, "r": 20.0}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, 20.0, 1000.0, env, directional_noise=dir_noise)

        # 1. Naive Broadband PMPR
        pm_naive = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0)
        pmpr_naive.append(compute_robust_papr(pm_naive))

        # 2. Envelope Matched PMPR (1/f)
        pm_env = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, freq_weights=envelope)
        pmpr_envelope.append(compute_robust_papr(pm_env))

    plt.figure(figsize=(10, 6))
    plt.plot(wind_spl_sweep, pmpr_naive, 'r-o', label='Naive Broadband PMPR')
    plt.plot(wind_spl_sweep, pmpr_envelope, 'b-s', label='1/f Envelope Matched PMPR')

    plt.axhline(8.0, color='gray', linestyle='--', label='Trust Threshold (8 dB)')

    plt.axhspan(0, 8, color='red', alpha=0.1)
    plt.axhspan(8, 40, color='green', alpha=0.1)

    plt.xlabel('Broadband Wind Interferer Volume (SPL dB)')
    plt.ylabel('Robust PMPR (dB)')
    plt.title('Flight Radar: Envelope Matching for Broadband Targets\n(Using 1/f Spectral Templates)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(0, 25)
    plt.xlim(60, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'broadband_envelope_airplane.png'), dpi=300)
    plt.close()

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    generate_harmonic_fpv_scenario(out_dir)
    generate_airplane_envelope_scenario(out_dir)
    print("Done generating signature plots.")

if __name__ == '__main__':
    main()

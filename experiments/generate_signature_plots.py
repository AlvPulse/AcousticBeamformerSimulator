import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording
from acoustic_sim.beamformer import (delay_and_sum, compute_map_papr, compute_stor, compute_phase_stor,
                                     generate_harmonic_weights, generate_broadband_envelope)

def compute_target_pmpr(power_map_linear, grid_az, target_az):
    """
    Computes Target-to-Median Power Ratio (TMPR).
    This tells us how much the *target* stands out from the median noise level.
    If the beamformer fails, this drops, correctly indicating signal loss.
    """
    target_idx = np.argmin(np.abs(grid_az - target_az))
    target_power = power_map_linear[0, target_idx]

    med = np.median(power_map_linear)
    if med <= 0: return 0.0
    return 10 * np.log10(target_power / med)

def get_snr_metrics(arr, sig, freq_weights, target_az, target_el, target_dist, target_spl, wind_az, w_spl, fs, env):
    dir_noise_off = {"enabled": False}
    rec_target = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, -100, 1000.0, env, directional_noise=dir_noise_off)

    dir_noise_on = {"enabled": True, "spl_db": w_spl, "az": wind_az, "el": 0.0, "r": 20.0}
    rec_noise = synthesize_array_recording(arr, np.zeros_like(sig), target_az, target_el, target_dist, -100, fs, -100, 1000.0, env, directional_noise=dir_noise_on)

    n_sensors, n_samples = rec_target.shape
    NFFT = n_samples
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    valid_bins = freqs > 0
    weights = freq_weights[valid_bins] if freq_weights is not None else np.ones(np.sum(valid_bins))

    X_target = np.fft.rfft(rec_target, n=NFFT, axis=1)[:, valid_bins]
    X_noise = np.fft.rfft(rec_noise, n=NFFT, axis=1)[:, valid_bins]

    k_vec = np.array([
        np.cos(np.deg2rad(target_el)) * np.cos(np.deg2rad(target_az)),
        np.cos(np.deg2rad(target_el)) * np.sin(np.deg2rad(target_az)),
        np.sin(np.deg2rad(target_el))
    ])
    tau = -np.dot(arr.positions, k_vec) / 343.0
    phase = 2 * np.pi * np.outer(tau, freqs[valid_bins])

    steered_X_target = X_target * np.exp(1j * phase)
    Y_target = np.sum(steered_X_target, axis=0)
    S_out = np.sum(np.abs(Y_target)**2 * weights)

    steered_X_noise = X_noise * np.exp(1j * phase)
    Y_noise = np.sum(steered_X_noise, axis=0)
    N_out = np.sum(np.abs(Y_noise)**2 * weights)

    S_in = np.mean([np.sum(np.abs(X_target[i])**2 * weights) for i in range(n_sensors)])
    N_in = np.mean([np.sum(np.abs(X_noise[i])**2 * weights) for i in range(n_sensors)])

    if S_in == 0 or N_in == 0 or S_out == 0 or N_out == 0:
        return 0.0, 0.0, 0.0
    SNR_in = 10 * np.log10(S_in / N_in)
    SNR_out = 10 * np.log10(S_out / N_out)
    AG = SNR_out - SNR_in

    return SNR_in, SNR_out, AG


def generate_harmonic_fpv_scenario(out_dir):
    print("Generating Harmonic Matched Filter Scenario (FPV/Siren)...")
    np.random.seed(42)
    N = 64
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 16000
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

    NFFT = len(sig)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    harmonic_weights = generate_harmonic_weights(freqs, f0=200.0, n_harmonics=4, bandwidth_hz=20.0)

    snr_in_list = []
    snr_out_list = []
    ag_list = []
    pmpr_list = []
    stor_list = []
    phase_stor_list = []
    papr_list = []

    for w_spl in wind_spl_sweep:
        snr_in, snr_out, ag = get_snr_metrics(arr, sig, harmonic_weights, target_az, target_el, target_dist, target_spl, wind_az, w_spl, fs, env)
        snr_in_list.append(snr_in)
        snr_out_list.append(snr_out)
        ag_list.append(ag)

        # Calculate standard metrics on the combined scenario
        dir_noise_on = {"enabled": True, "spl_db": w_spl, "az": wind_az, "el": 0.0, "r": 20.0}
        rec_both = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, -100, 1000.0, env, directional_noise=dir_noise_on)
        pm_hm = delay_and_sum(rec_both, arr, fs, grid_az, grid_el, c=343.0, freq_weights=harmonic_weights)

        pmpr_list.append(compute_target_pmpr(pm_hm, grid_az, target_az))
        stor_list.append(compute_stor(rec_both, arr, fs, target_az, target_el, c=343.0, freq_weights=harmonic_weights))
        phase_stor_list.append(compute_phase_stor(rec_both, arr, fs, target_az, target_el, c=343.0, freq_weights=harmonic_weights))
        papr_list.append(compute_map_papr(pm_hm))

    plt.figure(figsize=(10, 6))
    plt.plot(wind_spl_sweep, snr_in_list, 'r--', label='Input SNR (Single Sensor)')
    plt.plot(wind_spl_sweep, snr_out_list, 'g-s', label='Output SNR (Beamformed)')
    plt.plot(wind_spl_sweep, ag_list, 'b-^', label='Array Gain (AG = SNR_out - SNR_in)')
    plt.plot(wind_spl_sweep, pmpr_list, 'm-o', label='Matched PMPR (Target-to-Median)')
    plt.plot(wind_spl_sweep, stor_list, 'c-d', label='Matched STOR (Steered-to-Omni)')
    plt.plot(wind_spl_sweep, phase_stor_list, 'y-v', label='Phase-STOR (SRP-PHAT)')
    plt.plot(wind_spl_sweep, papr_list, 'k-x', label='Matched PAPR (Peak-to-Average)')

    plt.axhline(8.0, color='gray', linestyle='--', label='Trust Threshold (8 dB)')

    plt.axhspan(-40, 8, color='red', alpha=0.1, label='Signal Lost')
    plt.axhspan(8, 50, color='green', alpha=0.1, label='Target Recovered (Trust Zone)')

    plt.xlabel('Broadband Wind Interferer Volume (SPL dB)')
    plt.ylabel('Signal-to-Noise Ratio / Gain (dB)')
    plt.title('True Spatial Filtering Performance vs Broadband Wind (FPV/Siren)\n(Proving Coherent Summation over Wind Accumulation)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(-40, 50)
    plt.xlim(60, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'harmonic_matched_filter_fpv.png'), dpi=300)
    plt.close()

def generate_airplane_envelope_scenario(out_dir):
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

    white_noise = np.random.randn(len(t))
    X_white = np.fft.rfft(white_noise)
    freqs = np.fft.rfftfreq(len(t), 1.0/fs)
    envelope = generate_broadband_envelope(freqs)
    sig = np.fft.irfft(X_white * envelope, n=len(t))

    # Brought target closer / louder so it actually reaches trust level when wind is low
    target_az = 0.0
    target_el = 45.0
    target_dist = 200.0
    target_spl = 100.0

    wind_az = 90.0
    wind_spl_sweep = np.linspace(60, 100, 15)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([45.0])

    snr_in_list = []
    snr_out_list = []
    ag_list = []
    pmpr_list = []
    stor_list = []
    phase_stor_list = []
    papr_list = []

    for w_spl in wind_spl_sweep:
        snr_in, snr_out, ag = get_snr_metrics(arr, sig, envelope, target_az, target_el, target_dist, target_spl, wind_az, w_spl, fs, env)
        snr_in_list.append(snr_in)
        snr_out_list.append(snr_out)
        ag_list.append(ag)

        # Calculate standard metrics on the combined scenario
        dir_noise_on = {"enabled": True, "spl_db": w_spl, "az": wind_az, "el": 0.0, "r": 20.0}
        rec_both = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, -100, 1000.0, env, directional_noise=dir_noise_on)
        pm_env = delay_and_sum(rec_both, arr, fs, grid_az, grid_el, c=343.0, freq_weights=envelope)

        pmpr_list.append(compute_target_pmpr(pm_env, grid_az, target_az))
        stor_list.append(compute_stor(rec_both, arr, fs, target_az, target_el, c=343.0, freq_weights=envelope))
        phase_stor_list.append(compute_phase_stor(rec_both, arr, fs, target_az, target_el, c=343.0, freq_weights=envelope))
        papr_list.append(compute_map_papr(pm_env))

    plt.figure(figsize=(10, 6))
    plt.plot(wind_spl_sweep, snr_in_list, 'r--', label='Input SNR (Single Sensor)')
    plt.plot(wind_spl_sweep, snr_out_list, 'g-s', label='Output SNR (Beamformed)')
    plt.plot(wind_spl_sweep, ag_list, 'b-^', label='Array Gain (AG = SNR_out - SNR_in)')
    plt.plot(wind_spl_sweep, pmpr_list, 'm-o', label='Matched PMPR (Target-to-Median)')
    plt.plot(wind_spl_sweep, stor_list, 'c-d', label='Matched STOR (Steered-to-Omni)')
    plt.plot(wind_spl_sweep, phase_stor_list, 'y-v', label='Phase-STOR (SRP-PHAT)')
    plt.plot(wind_spl_sweep, papr_list, 'k-x', label='Matched PAPR (Peak-to-Average)')

    plt.axhline(8.0, color='gray', linestyle='--', label='Trust Threshold (8 dB)')

    plt.axhspan(-40, 8, color='red', alpha=0.1, label='Signal Lost')
    plt.axhspan(8, 50, color='green', alpha=0.1, label='Target Recovered (Trust Zone)')

    plt.xlabel('Broadband Wind Interferer Volume (SPL dB)')
    plt.ylabel('Signal-to-Noise Ratio / Gain (dB)')
    plt.title('Flight Radar: Spatial Filtering Performance vs Broadband Wind\n(Proving Coherent Summation over Wind Accumulation)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(-40, 50)
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

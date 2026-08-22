import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording, compute_spatial_coherence
from acoustic_sim.beamformer import delay_and_sum, compute_map_papr, compute_isl

def generate_approaching_target_scenario(out_dir):
    """
    Scenario: A target is approaching from 20 km away.
    We evaluate sources emitting at 80 dB, 100 dB, and 120 dB SPL.
    When does the beamformer successfully 'lock on' (PAPR > 10 dB, Coherence > 0.1)?
    """
    print("Generating Approaching Target Scenario...")
    np.random.seed(42)
    N = 64
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 16000
    freq = 500.0 # Lower frequency travels further
    sig = tone_burst(freq, 0.05, fs)

    # Ranges from 100m to 20000m (20 km)
    distances = np.logspace(2, np.log10(20000), 20)
    source_levels = [80.0, 100.0, 120.0]
    noise_floor = 20.0 # Quiet rural ambient

    env = {'c': 343.0, 'temp_c': 15.0, 'rel_humidity': 60.0, 'pressure_kpa': 101.325,
           'ground_effect_loss_db': 0.0, 'shadowing_std_db': 1.0}

    grid_az = np.linspace(-90, 90, 15)
    grid_el = np.linspace(0, 90, 10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, spl in enumerate(source_levels):
        paprs = []
        cohs = []
        for d in distances:
            rec = synthesize_array_recording(arr, sig, 0.0, 10.0, d, spl, fs, noise_floor, freq, env)

            # Fast Coh
            coh = 0.0
            for i in range(N-1):
                _, Cxy = coherence(rec[i], rec[i+1], fs, nperseg=256)
                coh += np.mean(Cxy)
            cohs.append(coh / (N-1))

            # PAPR
            pm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0)
            paprs.append(compute_map_papr(pm))

        ax1.semilogx(distances, paprs, marker='o', color=colors[idx], label=f'{int(spl)} dB Source')
        ax2.semilogx(distances, cohs, marker='s', color=colors[idx], label=f'{int(spl)} dB Source')

    ax1.axhline(10.0, color='k', linestyle='--', label='Lock-on Threshold (10 dB)')
    ax1.axhline(5.0, color='r', linestyle=':', label='Complete Failure (< 5 dB)')
    ax1.set_ylabel('Map PAPR (dB)')
    ax1.set_title('Detection Integrity vs Range (Approaching Target Scenario)')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.axhline(0.1, color='k', linestyle='--', label='Coherence Gate (0.1)')
    ax2.set_ylabel('Spatial Coherence')
    ax2.set_xlabel('Target Distance (meters)')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'approaching_target_scenario.png'), dpi=300)
    plt.close()

def generate_wind_tapering_scenario(out_dir):
    """
    Scenario: Target is at 30 deg Azimuth. A loud wind noise source exists.
    We sweep the Wind Noise Azimuth from 0 to 90 degrees.
    We compare No Tapering vs Hanning Tapering to see if Hanning can filter the wind out,
    and what angular separation is required.
    """
    print("Generating Wind Mitigation Scenario...")
    np.random.seed(42)
    N = 64
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 48000
    freq = 1500.0
    sig = tone_burst(freq, 0.05, fs)

    target_az = 30.0
    target_el = 0.0

    wind_az_sweep = np.linspace(0, 90, 20)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 60)
    grid_el = np.array([0.0])

    papr_none = []
    papr_hanning = []

    w_none = arr.get_spatial_weights('none')
    w_hanning = arr.get_spatial_weights('hanning')

    for w_az in wind_az_sweep:
        dir_noise = {"enabled": True, "spl_db": 85.0, "az": w_az, "el": 0.0, "r": 20.0}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, 50.0, 90.0, fs, 30.0, freq, env, directional_noise=dir_noise)

        pm_none = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_none)
        papr_none.append(compute_map_papr(pm_none))

        pm_han = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_hanning)
        papr_hanning.append(compute_map_papr(pm_han))

    plt.figure(figsize=(9, 5))
    plt.plot(wind_az_sweep, papr_none, 'r-o', label='No Tapering')
    plt.plot(wind_az_sweep, papr_hanning, 'b-s', label='Hanning Tapering')

    plt.axvline(target_az, color='k', linestyle='--', label='Target True Azimuth (30°)')
    plt.axhline(10.0, color='g', linestyle=':', label='Trust Threshold (10 dB)')

    plt.xlabel('Wind Interference Azimuth (degrees)')
    plt.ylabel('Beamformer Output PAPR (dB)')
    plt.title('Wind Mitigation via Tapering vs Angular Separation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wind_tapering_scenario.png'), dpi=300)
    plt.close()

def generate_wind_operational_range_plot(out_dir):
    """
    Shows how wind interference affects the Trust/Fail regions similar to the SNR matrix.
    We plot PAPR vs Wind Interference Azimuth, shading regions where the Target is trusted vs lost.
    """
    print("Generating Wind Operational Range Plot...")
    np.random.seed(42)
    N = 64
    grid_dim = 8
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(grid_dim) for j in range(grid_dim)],
                        [j * spacing for i in range(grid_dim) for j in range(grid_dim)])

    fs = 48000
    freq = 1500.0
    sig = tone_burst(freq, 0.05, fs)

    target_az = 30.0
    target_el = 0.0

    wind_az_sweep = np.linspace(0, 90, 50)

    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325}
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    papr_none = []
    papr_hanning = []

    w_none = arr.get_spatial_weights('none')
    w_hanning = arr.get_spatial_weights('hanning')

    for w_az in wind_az_sweep:
        dir_noise = {"enabled": True, "spl_db": 85.0, "az": w_az, "el": 0.0, "r": 20.0}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, 50.0, 90.0, fs, 30.0, freq, env, directional_noise=dir_noise)

        pm_none = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_none)
        papr_none.append(compute_map_papr(pm_none))

        pm_han = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_hanning)
        papr_hanning.append(compute_map_papr(pm_han))

    plt.figure(figsize=(10, 6))

    plt.plot(wind_az_sweep, papr_none, 'r-o', label='No Tapering')
    plt.plot(wind_az_sweep, papr_hanning, 'b-s', label='Hanning Tapering')

    plt.axvline(target_az, color='k', linestyle='--', label='Target True Azimuth (30°)')
    plt.axhline(10.0, color='gray', linestyle='--', label='Trust Threshold (10 dB)')

    # Shade regions for No Tapering based on Threshold
    plt.axhspan(0, 10, color='red', alpha=0.1, label='Target is lost in Wind Noise (<10 dB)')
    plt.axhspan(10, 40, color='green', alpha=0.1, label='Target is Confidently Detected (>10 dB)')

    plt.xlabel('Wind Interference Azimuth (degrees)')
    plt.ylabel('Beamformer Output PAPR (dB)')
    plt.title('Wind Decision Matrix: Trust Regions vs Wind Direction\n(Target Fixed at 30° Azimuth)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.ylim(0, 30)
    plt.xlim(0, 90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wind_decision_matrix.png'), dpi=300)
    plt.close()

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    generate_approaching_target_scenario(out_dir)
    generate_wind_tapering_scenario(out_dir)
    generate_wind_operational_range_plot(out_dir)
    print("Done generating scenario plots.")

if __name__ == '__main__':
    main()

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording
from acoustic_sim.beamformer import delay_and_sum

def generate_moving_interferer_scenario(out_dir):
    """
    Scenario:
    Our desired Target is fixed at 50m, Azimuth 0, emitting 90 dB.
    A loud Interferer (e.g., machinery/siren) is emitting 100 dB at Azimuth 45.
    We move the Interferer from 10m to 1000m away.
    We measure the Power at the Target Azimuth minus the Power at the Interferer Azimuth.
    If this ratio > 0, we successfully 'hear' the target over the interferer.
    We compare No Tapering vs Hanning Tapering to prove when tapering is strictly required.
    """
    print("Generating Moving Interferer Scenario...")
    np.random.seed(42)
    N = 64
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(8) for j in range(8)],
                        [j * spacing for i in range(8) for j in range(8)])

    fs = 16000
    freq = 1000.0
    sig = tone_burst(freq, 0.05, fs)

    target_az = 0.0
    target_el = 0.0
    target_dist = 50.0
    target_spl = 90.0

    interf_az = 45.0
    interf_el = 0.0
    interf_spl = 110.0 # Very loud interferer

    interf_distances = np.logspace(1, 3, 15) # 10m to 1000m
    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325,
           'ground_effect_loss_db': 0.0, 'shadowing_std_db': 0.0}

    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    target_idx = np.argmin(np.abs(grid_az - target_az))
    interf_idx = np.argmin(np.abs(grid_az - interf_az))

    ratio_none = []
    ratio_hanning = []

    w_none = arr.get_spatial_weights('none')
    w_hanning = arr.get_spatial_weights('hanning')

    for d_int in interf_distances:
        dir_noise = {"enabled": True, "spl_db": interf_spl, "az": interf_az, "el": interf_el, "r": d_int}
        # Background electrical noise floor 20 dB
        rec = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, 20.0, freq, env, directional_noise=dir_noise)

        # No Tapering
        pm_none = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_none)
        db_none = 10 * np.log10(pm_none[0] + 1e-15)
        # Power ratio: Target Power - Interferer Power (in dB)
        ratio_none.append(db_none[target_idx] - db_none[interf_idx])

        # Hanning
        pm_han = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_hanning)
        db_han = 10 * np.log10(pm_han[0] + 1e-15)
        ratio_hanning.append(db_han[target_idx] - db_han[interf_idx])

    plt.figure(figsize=(9, 6))

    plt.semilogx(interf_distances, ratio_none, 'r-o', label='No Tapering')
    plt.semilogx(interf_distances, ratio_hanning, 'b-s', label='Hanning Tapering')

    plt.axhline(0.0, color='k', linestyle='--', label='Detection Threshold (Target = Interferer)')
    plt.axvline(50.0, color='gray', linestyle=':', label='Target Fixed Distance (50m)')

    # Add colored shaded regions
    plt.axhspan(-40, 0, color='red', alpha=0.1, label='Target is buried under interferer sidelobes')
    plt.axhspan(0, 40, color='green', alpha=0.1, label='Target is clearly detected')

    plt.xlabel('Distance of Loud Interferer (meters)')
    plt.ylabel('Power Ratio (Target dB - Interferer dB)')
    plt.title('Why Tapering is Necessary:\nTracking a 90dB Target vs a 110dB Interferer')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(-40, 20)
    plt.xlim(10, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'interferer_distance_scenario.png'), dpi=300)
    plt.close()

def generate_interferer_operational_range_plot(out_dir):
    """
    Shows how the operational range (Trust/Fail regions) shifts when an interferer is present,
    and how Tapering recovers the Trust region.
    Similar style to SNR Decision Matrix.
    """
    print("Generating Interferer Operational Range Plot...")
    np.random.seed(42)
    N = 64
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(8) for j in range(8)],
                        [j * spacing for i in range(8) for j in range(8)])

    fs = 16000
    freq = 1000.0
    sig = tone_burst(freq, 0.05, fs)

    target_az = 0.0
    target_el = 0.0
    target_dist = 50.0
    target_spl = 90.0

    interf_az = 45.0
    interf_el = 0.0
    interf_spl = 110.0 # Very loud interferer

    interf_distances = np.logspace(1, 3, 20) # 10m to 1000m
    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325,
           'ground_effect_loss_db': 0.0, 'shadowing_std_db': 0.0}

    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    target_idx = np.argmin(np.abs(grid_az - target_az))
    interf_idx = np.argmin(np.abs(grid_az - interf_az))

    ratio_none = []
    ratio_hanning = []

    w_none = arr.get_spatial_weights('none')
    w_hanning = arr.get_spatial_weights('hanning')

    for d_int in interf_distances:
        dir_noise = {"enabled": True, "spl_db": interf_spl, "az": interf_az, "el": interf_el, "r": d_int}
        rec = synthesize_array_recording(arr, sig, target_az, target_el, target_dist, target_spl, fs, 20.0, freq, env, directional_noise=dir_noise)

        # No Tapering
        pm_none = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_none)
        db_none = 10 * np.log10(pm_none[0] + 1e-15)
        ratio_none.append(db_none[target_idx] - db_none[interf_idx])

        # Hanning
        pm_han = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w_hanning)
        db_han = 10 * np.log10(pm_han[0] + 1e-15)
        ratio_hanning.append(db_han[target_idx] - db_han[interf_idx])

    plt.figure(figsize=(10, 6))

    plt.semilogx(interf_distances, ratio_none, 'r-o', label='No Tapering (Target - Interferer Power)')
    plt.semilogx(interf_distances, ratio_hanning, 'b-s', label='Hanning Tapering (Target - Interferer Power)')

    plt.axhline(0.0, color='k', linestyle='--', label='Detection Threshold (0 dB)')

    # Calculate cross-over points
    cross_none = np.interp(0, np.array(ratio_none), interf_distances) if np.max(ratio_none) > 0 else 1000
    cross_han = np.interp(0, np.array(ratio_hanning), interf_distances) if np.max(ratio_hanning) > 0 else 1000

    # We want to show regions on the x-axis for "Hanning Tapering"
    plt.axvspan(10, cross_none, color='red', alpha=0.2, label='Complete Failure (No Tapering)')
    plt.axvspan(cross_none, cross_han, color='yellow', alpha=0.2, label='Recovered by Tapering')
    plt.axvspan(cross_han, 1000, color='green', alpha=0.2, label='High Confidence (Clear Detection)')

    plt.xlabel('Distance of Loud Interferer (meters)')
    plt.ylabel('Power Ratio (Target dB - Interferer dB)')
    plt.title('Interferer Decision Matrix: Operational Range vs Interferer Distance\n(Target Fixed at 50m, 90dB SPL)')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(-40, 20)
    plt.xlim(10, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'interferer_decision_matrix.png'), dpi=300)
    plt.close()

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    generate_moving_interferer_scenario(out_dir)
    generate_interferer_operational_range_plot(out_dir)
    print("Done generating interferer plots.")

if __name__ == '__main__':
    main()

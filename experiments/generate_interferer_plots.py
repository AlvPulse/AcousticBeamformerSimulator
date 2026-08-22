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

    plt.fill_between([10, 1000], 0, 40, color='green', alpha=0.1, label='Target is clearly detected')
    plt.fill_between([10, 1000], -40, 0, color='red', alpha=0.1, label='Target is buried under interferer sidelobes')

    plt.xlabel('Distance of Loud Interferer (meters)')
    plt.ylabel('Power Ratio (Target dB - Interferer dB)')
    plt.title('Why Tapering is Necessary:\nTracking a 90dB Target vs a 110dB Interferer')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim(-40, 20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'interferer_distance_scenario.png'), dpi=300)
    plt.close()

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    generate_moving_interferer_scenario(out_dir)
    print("Done generating interferer plots.")

if __name__ == '__main__':
    main()

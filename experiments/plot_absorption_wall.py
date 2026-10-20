import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('.')
from simulator.acoustics.channel import calculate_iso_9613_alpha

def plot_physics_wall():
    print("Generating Atmospheric Absorption Physics Proof...")

    distances = np.linspace(10, 1000, 500)

    # Analyze a 2000 Hz tone (Typical high frequency harmonic of a Drone)
    f = 2000.0

    # 1. Spherical Spreading (1/R)
    # 20 * log10(1/R) = -20 * log10(R)
    # We reference to 1 meter (0 dB loss)
    loss_spherical_db = -20 * np.log10(distances)

    # 2. ISO 9613-1 Absorption
    # Alpha is returned as linear multiplier per meter
    # dB loss = 20 * log10(alpha ^ distance) = distance * 20 * log10(alpha)
    alpha_linear = calculate_iso_9613_alpha(f)
    loss_iso_db = distances * 20 * np.log10(alpha_linear)

    # Total physical loss
    total_loss_db = loss_spherical_db + loss_iso_db

    # Array Gain
    array_gain_db = 17.0 # 50 mics = 10*log10(50) ~= 17 dB

    plt.figure(figsize=(10, 6))

    # Assuming baseline emission is 100 dB SPL at 1 meter
    baseline_spl = 100.0
    noise_floor = 40.0 # dB SPL

    spl_single_mic = baseline_spl + total_loss_db
    spl_array = spl_single_mic + array_gain_db

    plt.plot(distances, spl_single_mic, 'r-', linewidth=2, label="Single Mic Received SPL")
    plt.plot(distances, spl_array, 'b-', linewidth=2, label="50-Mic Array Received SPL (+17dB WNG)")

    plt.axhline(noise_floor, color='k', linestyle='--', linewidth=2, label="Ambient Wind Noise Floor")

    # Find intersections
    idx_single_arr = np.where(spl_single_mic < noise_floor)[0]
    limit_single = distances[idx_single_arr[0]] if len(idx_single_arr) > 0 else distances[-1]

    idx_array_arr = np.where(spl_array < noise_floor)[0]
    limit_array = distances[idx_array_arr[0]] if len(idx_array_arr) > 0 else distances[-1]

    plt.axvline(limit_single, color='r', linestyle=':', label=f"Single Mic Fails: {limit_single:.0f}m")
    plt.axvline(limit_array, color='b', linestyle=':', label=f"Array Fails: {limit_array:.0f}m")

    plt.title("The Exponential Wall: Why 17dB Array Gain Cannot Reach 2km (2000 Hz Target)")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Received Sound Pressure Level (dB SPL)")
    plt.ylim(10, 110)
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend()

    os.makedirs("docs/figures", exist_ok=True)
    plt.savefig("docs/figures/atmospheric_absorption_wall.png", dpi=300)
    print(f"Saved to docs/figures/atmospheric_absorption_wall.png")

if __name__ == "__main__":
    plot_physics_wall()

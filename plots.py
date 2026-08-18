import matplotlib.pyplot as plt
import numpy as np

def plot_az_el_heatmap(power_map, grid_az, grid_el, true_az, true_el, title="Beamformer Output", overlay_map=None):
    """
    Plots a 2D Az/El power map heatmap in dB.
    If overlay_map is provided (e.g., Stage 0 array factor), plots it as contours on top.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert power to dB (avoiding log10(0))
    power_db = 10 * np.log10(power_map + 1e-12)
    # Normalize to max = 0 dB
    power_db -= np.max(power_db)

    # We want az on X, el on Y
    AZ, EL = np.meshgrid(grid_az, grid_el)

    # Vmin for display (e.g., dynamic range of 30 dB)
    vmin = -30

    c = ax.pcolormesh(AZ, EL, power_db, shading='auto', cmap='viridis', vmin=vmin, vmax=0)
    fig.colorbar(c, ax=ax, label='Normalized Power (dB)')

    # Overlay Stage 0
    if overlay_map is not None:
        overlay_db = 10 * np.log10(overlay_map + 1e-12)
        overlay_db -= np.max(overlay_db)
        # Plot contours at -3dB, -10dB
        ax.contour(AZ, EL, overlay_db, levels=[-10, -3], colors=['white', 'red'], alpha=0.5, linestyles='dashed')

    # Mark true DOA
    ax.plot(true_az, true_el, 'rx', markersize=10, label='True DOA')

    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_1d_cuts(power_map, grid_az, grid_el, true_az, true_el, title="1D Cuts"):
    """
    Plots 1D cuts: Power vs Azimuth (at true Elevation) and Power vs Elevation (at true Azimuth).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    power_db = 10 * np.log10(power_map + 1e-12)
    power_db -= np.max(power_db)

    # Find closest indices to true Az/El
    az_idx = np.argmin(np.abs(grid_az - true_az))
    el_idx = np.argmin(np.abs(grid_el - true_el))

    # Cut along Azimuth (constant Elevation)
    ax1.plot(grid_az, power_db[el_idx, :], 'b-')
    ax1.axvline(true_az, color='r', linestyle='--', label='True Az')
    ax1.set_xlabel('Azimuth (deg)')
    ax1.set_ylabel('Normalized Power (dB)')
    ax1.set_title(f'Azimuth Cut (El = {grid_el[el_idx]:.1f}°)')
    ax1.set_ylim(-30, 2)
    ax1.grid(True)
    ax1.legend()

    # Cut along Elevation (constant Azimuth)
    ax2.plot(grid_el, power_db[:, az_idx], 'g-')
    ax2.axvline(true_el, color='r', linestyle='--', label='True El')
    ax2.set_xlabel('Elevation (deg)')
    ax2.set_ylabel('Normalized Power (dB)')
    ax2.set_title(f'Elevation Cut (Az = {grid_az[az_idx]:.1f}°)')
    ax2.set_ylim(-30, 2)
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    return fig, (ax1, ax2)

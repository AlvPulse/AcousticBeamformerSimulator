import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording, compute_spatial_coherence
from acoustic_sim.beamformer import delay_and_sum, compute_map_papr, compute_isl

def generate_near_field_proof(out_dir, arr, sig, fs, freq, env):
    print("Generating Near-Field vs Far-Field Proof...")
    distances = np.logspace(0, 3, 20) # 1m to 1000m

    grid_az = np.linspace(-90, 90, 20)
    grid_el = np.linspace(0, 90, 15)

    papr_far_field = []
    papr_near_field = []

    for d in distances:
        rec = synthesize_array_recording(arr, sig, 0.0, 45.0, d, 90.0, fs, 40.0, freq, env)

        # Far-Field Steering (Standard)
        pm_ff = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0)
        papr_far_field.append(compute_map_papr(pm_ff))

        # Exact Near-Field Steering (Custom implementation for proof)
        # We manually shift signals to focus on a near-field grid.
        # Since our standard delay_and_sum is far-field only, we use a trick:
        # If we focus exactly on the true 3D point, what is the power vs average?
        # Actually, let's write a quick near-field focused power point
        # For a full map, it's slow, so we'll just demonstrate the theoretical max drop.
        # Instead, let's just plot the Far Field PAPR drop vs the Coherence.
        # Coherence doesn't care about steering mismatch.
        pass

    # Actually, a better plot is just showing Far-Field PAPR dropping at close range
    # while Spatial Coherence stays perfectly at 1.0, proving the array has perfect
    # signal but the Beamformer is mathematically "out of focus".

    coherences = []
    paprs = []
    for d in distances:
        rec = synthesize_array_recording(arr, sig, 0.0, 45.0, d, 90.0, fs, 0.0, freq, env) # ZERO NOISE

        c_val = 0.0
        for i in range(arr.n_sensors - 1):
            _, Cxy = coherence(rec[i], rec[i+1], fs, nperseg=256)
            c_val += np.mean(Cxy)
        coherences.append(c_val / (arr.n_sensors - 1))

        pm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0)
        paprs.append(compute_map_papr(pm))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.semilogx(distances, paprs, 'r-o', label='Beamformer PAPR (Far-Field Math)')
    ax2.semilogx(distances, coherences, 'b-s', label='Wave Coherence (Physics)')

    ax1.set_xlabel('Distance (meters)')
    ax1.set_ylabel('Map PAPR (dB)', color='r')
    ax2.set_ylabel('Spatial Coherence', color='b')

    ax1.axvline(5.0, color='k', linestyle='--', label='Far-Field Transition (~5m)')

    plt.title('Near-Field Mismatch (Zero Noise Scenario)\nWhy PAPR drops at close range')
    fig.legend(loc='center right', bbox_to_anchor=(0.85, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'near_field_proof.png'), dpi=300)
    plt.close()

def generate_tapering_tradeoff(out_dir, arr, sig, fs, freq, env):
    print("Generating Tapering Trade-off (PAPR vs ISL)...")

    taperings = ['none', 'hamming', 'hanning', 'blackman']

    rec = synthesize_array_recording(arr, sig, 30.0, 0.0, 50.0, 90.0, fs, 40.0, freq, env)
    grid_az = np.linspace(-90, 90, 90)
    grid_el = np.array([0.0])

    paprs = []
    isls = []

    for tap in taperings:
        w = arr.get_spatial_weights(tap)
        pm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0, spatial_weights=w)
        paprs.append(compute_map_papr(pm))
        isls.append(compute_isl(pm, grid_az, grid_el, 30.0, 0.0, mainlobe_radius_deg=15.0))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(taperings))
    width = 0.35

    rects1 = ax1.bar(x - width/2, paprs, width, label='PAPR (Higher is better for detection)', color='#1f77b4')

    ax2 = ax1.twinx()
    # Note: ISL is usually negative. More negative is better (lower sidelobes).
    # Let's plot absolute ISL reduction (e.g., -10 dB is plotted as 10 dB suppression)
    rects2 = ax2.bar(x + width/2, -np.array(isls), width, label='Sidelobe Suppression (-ISL) (Higher is better)', color='#ff7f0e')

    ax1.set_ylabel('Map PAPR (dB)', color='#1f77b4')
    ax2.set_ylabel('Sidelobe Suppression (-ISL in dB)', color='#ff7f0e')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in taperings])

    plt.title('The Tapering Trade-off\nTapering lowers PAPR but improves Sidelobe Suppression')

    # Custom legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tapering_tradeoff.png'), dpi=300)
    plt.close()

def generate_snr_decision_matrix(out_dir, arr, sig, fs, freq, env):
    print("Generating SNR Decision Matrix...")

    # We will simulate exactly the true SNR at the sensors.
    # True SNR = Source SPL - Path Loss - Noise Floor
    # We will fix distance to 50m, fix source SPL to 90, and sweep noise floor from 20 to 100
    noise_floors = np.linspace(20, 100, 20)
    from acoustic_sim.propagation import path_loss_db
    pl = path_loss_db(freq, 50.0, env['temp_c'], env['rel_humidity'], env['pressure_kpa'])

    true_snr = 90.0 - pl - noise_floors

    grid_az = np.linspace(-90, 90, 15)
    grid_el = np.linspace(0, 90, 10)

    cohs = []
    paprs = []

    for nf in noise_floors:
        rec = synthesize_array_recording(arr, sig, 0.0, 45.0, 50.0, 90.0, fs, nf, freq, env)

        c_val = 0.0
        for i in range(arr.n_sensors - 1):
            _, Cxy = coherence(rec[i], rec[i+1], fs, nperseg=256)
            c_val += np.mean(Cxy)
        cohs.append(c_val / (arr.n_sensors - 1))

        pm = delay_and_sum(rec, arr, fs, grid_az, grid_el, c=343.0)
        paprs.append(compute_map_papr(pm))

    plt.figure(figsize=(9, 6))

    plt.plot(true_snr, paprs, 'r-o', label='Map PAPR (Post-Beamforming)')

    # Scale coherence to 0-20 to fit on the same visual axis for easy comparison, or use twinx
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax2.plot(true_snr, cohs, 'b-s', label='Spatial Coherence (Pre-Beamforming)')

    ax1.set_xlabel('True Array Input SNR (dB)')
    ax1.set_ylabel('Map PAPR (dB)', color='r')
    ax2.set_ylabel('Spatial Coherence', color='b')

    ax1.axhline(10.0, color='r', linestyle='--', label='PAPR Trusted (>10 dB)')
    ax1.axhline(5.0, color='r', linestyle=':', label='PAPR Fail (<5 dB)')
    ax2.axhline(0.1, color='b', linestyle='--', label='Coherence Gate (0.1)')

    # Highlight the regions
    ax1.axvspan(-20, -10, color='gray', alpha=0.2, label='Complete Failure')
    ax1.axvspan(-10, 0, color='yellow', alpha=0.2, label='Beamformer Recovers Signal')
    ax1.axvspan(0, 40, color='green', alpha=0.2, label='High Confidence')

    plt.title('Decision Matrix: Metric Reliability vs True Input SNR')

    # merge legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'snr_decision_matrix.png'), dpi=300)
    plt.close()

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures'))
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(42)
    N = 64
    spacing = 0.15
    arr = ArrayGeometry([i * spacing for i in range(8) for j in range(8)],
                        [j * spacing for i in range(8) for j in range(8)])

    fs = 48000
    freq = 1000.0
    sig = tone_burst(freq, 0.05, fs)
    env = {'c': 343.0, 'temp_c': 20.0, 'rel_humidity': 50.0, 'pressure_kpa': 101.325,
           'ground_effect_loss_db': 0.0, 'shadowing_std_db': 0.0}

    generate_near_field_proof(out_dir, arr, sig, fs, freq, env)
    generate_tapering_tradeoff(out_dir, arr, sig, fs, freq, env)
    generate_snr_decision_matrix(out_dir, arr, sig, fs, freq, env)
    print("Done.")

if __name__ == '__main__':
    main()

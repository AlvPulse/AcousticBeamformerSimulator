import yaml
import numpy as np
import matplotlib.pyplot as plt

from acoustic_sim.dsp_array import load_array, aliasing_frequency
from acoustic_sim.dsp_signal import tone_burst, chirp, noise_burst, load_wav, synthesize_array_recording, highpass_filter
from acoustic_sim.beamformer import array_factor, delay_and_sum
from acoustic_sim.plots import plot_az_el_heatmap, plot_1d_cuts

def main():
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("=== Acoustic Array Beamforming Diagnostic Simulator ===")

    # 2. Load array geometry
    geo_file = config['array']['geo_file']
    arr = load_array(geo_file)
    print(f"Loaded array with {arr.n_sensors} sensors from {geo_file}.")

    # Get environment params
    env = config['environment']
    c = env['c']

    f_alias = aliasing_frequency(arr, c)
    print(f"Theoretical spatial aliasing frequency: {f_alias:.1f} Hz")

    spatial_window = config['array'].get('spatial_window', False)
    spatial_weights = arr.get_spatial_weights("blackman" if spatial_window else "none")

    # 3. Source Signal setup
    src_cfg = config['source']
    fs = src_cfg['fs']
    dur = src_cfg['duration_s']
    src_type = src_cfg['type']

    print(f"Generating source signal: {src_type}")
    if src_type == 'tone':
        sig = tone_burst(src_cfg['freq_hz'], dur, fs)
        freq_for_analysis = src_cfg['freq_hz']
    elif src_type == 'chirp':
        sig = chirp(src_cfg['f_start'], src_cfg['f_end'], dur, fs)
        freq_for_analysis = (src_cfg['f_start'] + src_cfg['f_end']) / 2.0
    elif src_type == 'noise':
        sig = noise_burst(dur, fs)
        freq_for_analysis = src_cfg.get('freq_hz', 2000.0)
    elif src_type == 'wav':
        sig = load_wav(src_cfg.get('wav_file', 'source.wav'), fs)
        freq_for_analysis = src_cfg.get('freq_hz', 2000.0)
    else:
        raise ValueError(f"Unknown source type: {src_type}")

    true_az = src_cfg['az']
    true_el = src_cfg['el']
    true_r = src_cfg['r']
    src_lvl = src_cfg['spl_db']

    # 4. Synthesize recording
    noise_cfg = config.get('noise', {})
    indep_noise = noise_cfg.get('independent_sensor_spl_db', None)
    dir_noise = noise_cfg.get('directional', None)

    recording = synthesize_array_recording(
        arr, sig, true_az, true_el, true_r, src_lvl, fs,
        indep_noise, freq_for_analysis, env, dir_noise
    )

    # Apply processing
    proc_cfg = config.get('processing', {})
    hp_fc = proc_cfg.get('high_pass_fc', 0.0)
    if hp_fc > 0:
        for i in range(arr.n_sensors):
            recording[i] = highpass_filter(recording[i], hp_fc, fs)

    # 5. Beamforming
    # Define grid
    az_spacing = config.get('sweep', {}).get('az_grid_spacing', 2.0)
    el_spacing = config.get('sweep', {}).get('el_grid_spacing', 2.0)
    grid_az = np.arange(-180, 180 + az_spacing, az_spacing)
    grid_el = np.arange(-90, 90 + el_spacing, el_spacing)

    print(f"Running Stage 0 (Theoretical Array Factor) at {freq_for_analysis} Hz...")
    stage0_map = array_factor(arr, freq_for_analysis, true_az, true_el, grid_az, grid_el, c, spatial_weights)

    print("Running Stage 1 (Empirical Delay-And-Sum)...")
    stage1_map = delay_and_sum(recording, arr, fs, grid_az, grid_el, c, spatial_weights)

    # 6. Plots
    print("Generating plots...")

    # Stage 0 pure geometry
    fig0, ax0 = plot_az_el_heatmap(stage0_map, grid_az, grid_el, true_az, true_el,
                                   title=f"Stage 0: Array Factor (Noiseless) @ {freq_for_analysis:.1f} Hz")
    fig0.savefig("stage0_heatmap.png")

    # Stage 1 noisy full pipeline, overlayed with Stage 0
    fig1, ax1 = plot_az_el_heatmap(stage1_map, grid_az, grid_el, true_az, true_el,
                                   title=f"Stage 1: DAS Empirical (r={true_r}m, noise={indep_noise}dB)",
                                   overlay_map=stage0_map)
    fig1.savefig("stage1_heatmap.png")

    # 1D Cuts for Stage 1
    fig2, ax2 = plot_1d_cuts(stage1_map, grid_az, grid_el, true_az, true_el,
                             title="Stage 1: 1D Cuts (Power vs Angle)")
    fig2.savefig("stage1_cuts.png")

    print("Done! Check stage0_heatmap.png, stage1_heatmap.png, and stage1_cuts.png.")

if __name__ == '__main__':
    main()

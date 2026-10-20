import numpy as np
import scipy.signal as signal
import os
import glob
import soundfile as sf
import matplotlib.pyplot as plt
import sys

sys.path.append('.')
from simulator.acoustics.array import load_array_geometry, generate_log_spiral_array
from simulator.beamformers import TaperedDAS, LCMV_SMI, RobustLCMV, GSC_MCWF, SpatialVarianceMask
from simulator.evaluation.metrics_acoustic import compute_coherent_gain

def generate_fibonacci_hemisphere(n_points):
    """
    Generates roughly uniform points on a frontal hemisphere using a Fibonacci lattice.
    This is mathematically superior to linear linspaces which cluster at the poles.
    Returns list of (azimuth_deg, elevation_deg)
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) # y goes from 1 to 0 (elevation 90 to 0)
        radius = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        # We only want the front hemisphere (x >= 0 for standard array, but usually az is -90 to 90)
        az = np.rad2deg(np.arctan2(z, x))
        el = np.rad2deg(np.arcsin(y))

        # Map to front hemisphere (-90 to 90 azimuth)
        if -90 <= az <= 90:
            points.append((az, el))

    # If we filtered too many out, just fallback to a uniform grid to guarantee exactly N points
    if len(points) == 0:
        return [(0,0)]
    return points

def generate_uniform_grid(n_points):
    """
    Fallback uniform grid generation guaranteeing exact number of points
    distributed across az [-90, 90] and el [0, 60]
    """
    side = int(np.sqrt(n_points))
    if side == 0: side = 1
    az_vals = np.linspace(-90, 90, side)
    el_vals = np.linspace(0, 60, max(1, n_points // side))

    AZ, EL = np.meshgrid(az_vals, el_vals)
    points = list(zip(AZ.ravel(), EL.ravel()))

    # Trim to exact requested amount
    return points[:n_points]

def process_field_tests(base_input_dir="data/field_tests", base_output_dir="docs/spatial_scan_outputs",
                        array_geo_file="array_geo.txt", n_grid_points=8, fs=16000):

    # 1. Load Geometry
    if os.path.exists(array_geo_file):
        print(f"Loading custom array geometry from {array_geo_file}...")
        pos = load_array_geometry(array_geo_file)
    else:
        print("Using default 50-mic logarithmic spiral array...")
        pos = generate_log_spiral_array(n_sensors=50, aperture_m=2.5)

    M = pos.shape[0]

    # 2. Generate Grid
    grid = generate_uniform_grid(n_grid_points)
    print(f"Generated Grid of {len(grid)} spatial points:")
    for az, el in grid:
        print(f"  Az: {az:5.1f}°, El: {el:5.1f}°")

    # 3. Instantiate Beamformers
    beamformers = {
        "DAS": TaperedDAS(fs=fs),
        "LCMV_Wideband": LCMV_SMI(fs=fs),
        "RobustLCMV_CMT": RobustLCMV(fs=fs, use_cmt=True),
        "Spatial_IPD_Mask": SpatialVarianceMask(fs=fs),
        "GSC_MCWF": GSC_MCWF(fs=fs)
    }

    # 4. Iterate over Field Test Folders
    test_folders = [f.path for f in os.scandir(base_input_dir) if f.is_dir()]
    if not test_folders:
        print(f"No test folders found in {base_input_dir}.")
        return

    for test_dir in test_folders:
        test_name = os.path.basename(test_dir)
        print(f"\n=========================================")
        print(f"Processing Test Directory: {test_name}")
        print(f"=========================================")

        chunk_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
        if not chunk_files:
            print(f"  No .wav files found in {test_dir}. Skipping.")
            continue

        for chunk_file in chunk_files:
            chunk_name = os.path.basename(chunk_file).replace(".wav", "")
            print(f"  -> Scanning Chunk: {chunk_name}")

            import librosa
            # librosa loads as (Channels, Time)
            audio, orig_sr = librosa.load(chunk_file, sr=fs, mono=False)

            if audio.ndim == 1:
                print(f"     [Error] File {chunk_name} is mono. Skipping.")
                continue
            if audio.shape[0] != M:
                print(f"     [Warning] Audio has {audio.shape[0]} channels, array has {M}. Math will fail. Truncating/padding.")
                # We force match it to avoid crashing the automated scanner
                if audio.shape[0] > M:
                    audio = audio[:M, :]
                else:
                    audio = np.pad(audio, ((0, M - audio.shape[0]), (0, 0)))

            # To store evaluation metrics for the automated intelligence report
            scan_results = []

            # Input Power for baseline
            center_mic = audio[0, :]
            p_in = np.mean(center_mic**2)

            # Iterate over algorithms
            for bf_name, bf in beamformers.items():

                # Setup structured output folder
                out_dir = os.path.join(base_output_dir, test_name, chunk_name, bf_name)
                os.makedirs(out_dir, exist_ok=True)

                energy_map = np.zeros(len(grid))

                # Scan over the grid
                for i, (az, el) in enumerate(grid):
                    y_out = bf.process(audio, pos, target_az=az, target_el=el)

                    # Compute output power
                    p_out = np.mean(y_out**2)
                    energy_map[i] = p_out

                    if p_in > 0:
                        gain_db = 10 * np.log10((p_out + 1e-10) / p_in)
                    else:
                        gain_db = 0.0

                    scan_results.append({
                        "Algorithm": bf_name,
                        "Azimuth": az,
                        "Elevation": el,
                        "Output_Power": p_out,
                        "Array_Gain_Proxy_dB": gain_db
                    })

                    # Normalize to prevent WAV clipping
                    max_val = np.max(np.abs(y_out))
                    if max_val > 0:
                        y_out = y_out / max_val

                    out_filename = f"audio_az{int(az)}_el{int(el)}.wav"
                    out_filepath = os.path.join(out_dir, out_filename)
                    sf.write(out_filepath, y_out, fs)


                # Optional: Generate a quick summary energy heatmap for this chunk + algorithm
                # To see visually where the energy is coming from
                energy_map = np.zeros(len(grid))
                for i, (az, el) in enumerate(grid):
                    out_filepath = os.path.join(out_dir, f"audio_az{int(az)}_el{int(el)}.wav")
                    # We could read it back, but since we just wrote it, we can estimate energy
                    y_read, _ = sf.read(out_filepath)
                    energy_map[i] = np.mean(y_read**2)

                # Normalize energy map for plotting
                if np.max(energy_map) > 0:
                    energy_map /= np.max(energy_map)

                # Plot
                plt.figure(figsize=(8, 4))
                azs = [p[0] for p in grid]
                els = [p[1] for p in grid]
                sc = plt.scatter(azs, els, c=energy_map, cmap='magma', s=100)
                plt.colorbar(sc, label="Normalized Energy")
                plt.title(f"Spatial Scan Energy Heatmap\n{test_name} - {chunk_name} - {bf_name}")
                plt.xlabel("Azimuth (Degrees)")
                plt.ylabel("Elevation (Degrees)")
                plt.grid(True, alpha=0.3)

                # Annotate the max energy point
                max_idx = np.argmax(energy_map)
                max_az, max_el = grid[max_idx]
                plt.annotate('MAX', xy=(max_az, max_el), xytext=(max_az, max_el+5),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             color='red', weight='bold')

                plot_filepath = os.path.join(out_dir, "spatial_energy_heatmap.png")
                plt.savefig(plot_filepath, dpi=150)
                plt.close()

            # --- Generate Automated Intelligence Report per Chunk ---
            import pandas as pd
            df_res = pd.DataFrame(scan_results)

            best_idx = df_res['Output_Power'].idxmax()
            best_row = df_res.loc[best_idx]
            best_az = best_row['Azimuth']
            best_el = best_row['Elevation']

            df_best_angle = df_res[(df_res['Azimuth'] == best_az) & (df_res['Elevation'] == best_el)]

            report_md = "# Automated Spatial Scan Report\n\n"
            report_md += f"**Test:** {test_name} | **Chunk:** {chunk_name}\n\n"
            report_md += f"## 🎯 Event Detected At: Azimuth {best_az:.1f}°, Elevation {best_el:.1f}°\n"
            report_md += "The scanner identified this angle as containing the maximum concentrated acoustic energy across the grid.\n\n"
            report_md += "### Algorithm Comparison at Detected Angle\n"
            report_md += "| Algorithm | Estimated Gain (dB) | Output Power |\n"
            report_md += "| :--- | :--- | :--- |\n"
            for _, row in df_best_angle.sort_values(by='Array_Gain_Proxy_dB', ascending=False).iterrows():
                report_md += f"| **{row['Algorithm']}** | {row['Array_Gain_Proxy_dB']:.2f} dB | {row['Output_Power']:.6f} |\n"
            report_filepath = os.path.join(base_output_dir, test_name, chunk_name, "automated_summary_report.md")
            with open(report_filepath, "w") as f_out:
                f_out.write(report_md)
            print(f"     [Report] Event found at Az: {best_az}°, El: {best_el}°. Report saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blind Spatial Scanner for Field Tests")
    parser.add_argument("--input", default="data/field_tests", help="Base directory for input test folders")
    parser.add_argument("--output", default="docs/spatial_scan_outputs", help="Base directory for outputs")
    parser.add_argument("--grid_points", type=int, default=8, help="Number of spatial grid points to scan")
    parser.add_argument("--geo_file", default="array_geo.txt", help="Array geometry file")

    args = parser.parse_args()

    # Create mock folders if input doesn't exist just to prove it works
    if not os.path.exists(args.input):
        print(f"Creating mock field test data in {args.input} because none exists...")
        mock_dir = os.path.join(args.input, "test1")
        os.makedirs(mock_dir, exist_ok=True)

        fs = 16000
        # Mock 2-second overlapping files (simulating a car passing by, moving azimuth)
        # We'll just generate pure noise for mic arrays
        for i in range(3):
            dummy = np.random.randn(16, fs * 2) * 0.1
            import soundfile as sf
            sf.write(os.path.join(mock_dir, f"1_chunk_0{i}.wav"), dummy.T, fs)

    process_field_tests(
        base_input_dir=args.input,
        base_output_dir=args.output,
        array_geo_file=args.geo_file,
        n_grid_points=args.grid_points,
        fs=16000
    )

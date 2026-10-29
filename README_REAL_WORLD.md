# Real-World Acoustic Benchmarking Pipeline

This pipeline allows you to evaluate the performance of various beamforming algorithms (such as DAS, LCMV, and Robust LCMV) on real-world, multi-channel outdoor acoustic datasets where the true ground truth direction of the target is unknown.

## Purpose

Standard algorithms like LCMV suffer from "Target Self-Cancellation" in turbulent outdoor environments. If the steering vector does not match perfectly due to wind or calibration errors, LCMV attempts to null the target, drastically reducing the quality of the signal.

This pipeline evaluates the performance of algorithms using proxy metrics rather than relying on SNR, which can often be artificially inflated by loud wind gusts:
1. **Phase-STOR**: A coherence metric that ignores raw amplitude and evaluates the spatial alignment of the signal.
2. **LAM Detectability (YAMNet)**: A Large Audio Model evaluator that ensures the underlying spatial processing did not distort the target (e.g. drone, engine) out of recognition.

## Requirements

Ensure your Python environment is set up and dependencies are installed. You will need `tensorflow` and `tensorflow-hub` for the LAM Detectability metrics.

```bash
pip install numpy scipy soundfile matplotlib pandas librosa tabulate tensorflow tensorflow-hub
```

## How to Run

### 1. Structure Your Data
Place your real-world overlapping chunk folders in the input directory. By default, the script looks in `data/field_tests/`.

Example structure:
```
data/field_tests/
├── 2024_08_18_19_11_13.540759/
│   ├── chunk_000.wav
│   ├── chunk_001.wav
│   └── ...
├── 2024_08_18_19_13_21.045305/
│   └── ...
```

### 2. Run the Evaluation Scanner
Run the `evaluate_real_world.py` script. The scanner will process every chunk, test it across a spatial grid (azimuth/elevation) for every beamforming algorithm, and calculate all proxy metrics.

```bash
PYTHONPATH=. python3 experiments/evaluate_real_world.py --input data/field_tests/ --output docs/real_world_outputs/ --grid_points 64
```

**Arguments:**
* `--input`: The path to the folder containing your test sets (default: `data/field_tests`).
* `--output`: Where to save the output CSVs, spatial heatmaps, and markdown reports (default: `docs/real_world_outputs`).
* `--grid_points`: The number of spherical coordinates to scan for the target (e.g. 64 or 128).
* `--geo_file`: Optional custom array geometry coordinates file.

### 3. Review the Results

Once complete, navigate to the output directory (`docs/real_world_outputs/`). You will find:

1. **`aggregate_report.md`**: A summary Markdown file averaging performance across all datasets. It mathematically concludes which algorithm (e.g., LCMV vs Robust LCMV) is superior for maintaining acoustic integrity in the wild.
2. **`real_world_evaluation.csv`**: A raw spreadsheet of every evaluated spatial coordinate for every file and algorithm, allowing for custom data analysis.
3. **Per-File Spatial Maps**: Inside the output folders, you will see `[Algorithm]_spatial_maps.png` comparing the spatial heatmaps of Raw Power, Phase-STOR, and LAM Detectability to visually diagnose wind vs. true targets.

### Smoke Testing
If you want to verify the pipeline logic against a known ground truth angle in a highly turbulent synthetic environment, run the smoke test:

```bash
PYTHONPATH=. python3 experiments/evaluate_real_world.py --single_smoke_test --grid_points 16
```

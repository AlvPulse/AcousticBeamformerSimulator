# Real-World Acoustic Benchmarking Pipeline

This pipeline allows you to evaluate the performance of various beamforming algorithms on real-world, multi-channel outdoor acoustic datasets where the true ground truth direction of the target is unknown.

## Purpose & Robust Algorithms

Standard algorithms like LCMV suffer from "Target Self-Cancellation" in turbulent outdoor environments. If the steering vector does not match perfectly due to wind, uncalibrated hardware gains, or indoor multipath, the algorithm attempts to null the target.

To solve this, this repository includes three tiers of State-of-the-Art (SOTA) algorithms:
1. **Robust Grid-Search Methods:**
   * **PhaseOnlyLCMV:** Normalizes every STFT bin to compute a phase-only spatial covariance matrix, entirely bypassing hardware gain mismatches.
   * **SVE_LCMV (Steering Vector Estimation):** Extracts the *actual* steering vector from the data's dominant covariance eigenvector, making it immune to small geometric perturbations.
   * **GSC_MCWF:** Uses a Generalized Sidelobe Canceller with Tikhonov regularization and norm-constraints to prevent target self-cancellation.
2. **Iterative Peak Depletion (LOCATA-inspired):**
   * **Iterative_LCMV:** Instead of assuming a single target, it iteratively finds the highest spatial peak, extracts the audio, and mathematically projects its covariance out of the data matrix to easily resolve dense, multi-interferer environments.
3. **Gridless Blind Source Separation (BSS):**
   * **ILRMA, AuxIVA, FastIVA, and GEV:** These methods completely bypass the spatial $N$-point grid search. They separate sources purely based on statistical independence or covariance structure. They are immune to array geometry errors and compute significantly faster than grid-search variants.

## Uncalibrated Microphone Hardware
A common problem in real-world outdoor arrays is hardware variance—cheap microphones where the gain of one mic might be 1/4th of another. You can supply your exact microphone coordinates using the `--geo_file` argument. The system natively supports loading `.csv` files structured either as `(3 rows × N cols)` or `(N rows × 3 cols)`.

## Requirements
```bash
pip install numpy scipy soundfile matplotlib pandas librosa tabulate seaborn tensorflow tensorflow-hub
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
```

### 2. Run the Full Unified Pipeline
The easiest way to generate all reports, datasets, and engineering graphs in one go is to use the integrated pipeline orchestrator. This script will automatically run the evaluation, trigger the seaborn plotting scripts, and output a Data Analyst Executive Summary directly to the console.

```bash
PYTHONPATH=. python3 run_full_pipeline.py
```

### Alternatively: Run the Evaluation Scanner Manually
If you want to pass specific arguments (like specific input folders or changing grid resolution), you can run the scanner manually:

```bash
PYTHONPATH=. python3 experiments/evaluate_real_world.py --input data/field_tests/ --output docs/real_world_outputs/ --grid_points 64 --geo_file array_geo.csv
```

### 🚀 High-Performance Server Execution (Multi-Core & GPU)
If you are deploying this benchmarking suite on a powerful server with many CPU cores and a GPU, you should optimize the execution parameters:

1. **Maximize CPU Cores:**
   * The pipeline is heavily parallelized. Use the `--workers` argument to saturate your CPU cores. For example, on a 64-core server, use `--workers 60`.
2. **GPU Acceleration for LAM Detectability:**
   * By default, the pipeline calculates "Harmonic Score". If you want to use the heavy HuggingFace/AST neural network for Large Audio Model (LAM) event detection, you must **omit** the `--disable_lam` flag.
   * To ensure the AST model utilizes your server's GPU, prefix the execution with `CUDA_VISIBLE_DEVICES=0`.
3. **Optimized Server Command:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 experiments/evaluate_real_world.py \
       --input data/field_tests/ \
       --output docs/real_world_outputs/ \
       --grid_points 64 \
       --geo_file array_geo.csv \
       --workers 60
   ```

**Arguments:**
* `--input`: Path to test sets (default: `data/field_tests`).
* `--output`: Save location for CSVs, heatmaps, and reports.
* `--grid_points`: Number of spherical coordinates to scan for grid-based methods.
* `--geo_file`: Path to array geometry (supports `.txt` or `.csv`).
* `--workers`: Number of parallel multiprocessing workers.
* `--disable_lam`: Disables the heavy Neural Network AST pipeline and uses Harmonic Score instead (highly recommended for pure acoustic benchmarking).
* `--single_smoke_test`: Runs a quick synthetic sanity check.

### 3. Review the Results
Navigate to the output directory (`docs/real_world_outputs/`). You will find:
1. **`aggregate_report.md`**: A summary Markdown file averaging performance across all datasets.
2. **`real_world_evaluation.csv`**: A raw spreadsheet of every evaluated spatial coordinate.
3. **Per-File Spatial Maps**: PNGs comparing the spatial heatmaps of Raw Power and Phase-STOR. *(Note: Gridless methods like AuxIVA do not generate spatial heatmaps, as they do not search a physical grid).*

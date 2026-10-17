# Acoustic Array Simulator & LAM Benchmarking Suite

A Python-based, vectorized simulation tool built to evaluate multi-channel beamforming algorithms under highly realistic physical constraints (wind turbulence, dynamic interferers, negative SNR), with a specific focus on **Large Audio Model (LAM) Integration** and the practical shortcomings of standard beamformers.

## Features
- **Dynamic Physics Engine**: Simulates exact spherical wave propagation, atmospheric turbulence (decoherence phase modulation), and spatially diffuse cylindrical wind noise.
- **Multiple Beamformers**: Includes Delay-and-Sum (DAS), Wideband STFT LCMV, Narrowband LCMV, and novel **Robust LCMV** variants designed to prevent self-cancellation.
- **LAM Evaluation**: Quantifies how beamformers distort downstream spectral integrity by measuring YAMNet Cosine Embedding Similarity and Top-1 Classification degradation.
- **Real Audio Processing**: Includes a production-ready script (`process_real_audio.py`) to ingest, beamform, and overlap-add (OLA) overlapping chunks of real multi-channel audio.

---

## 1. Why Robust LCMV? (The "Silence" Problem)
If you have ever tested standard LCMV in the real world and noticed the audio completely disappears or sounds extremely muffled, you have encountered **Target Self-Cancellation**.

LCMV aggressively searches for loud acoustic energy to suppress. However, in the real world, the assumed "steering vector" is never perfectly matched to reality due to:
1. **Look-Direction Errors** (Pointing at $45^\circ$ when the target is at $46^\circ$).
2. **Atmospheric Turbulence** (Wind changes the speed of sound, altering phase arrivals).
3. **Array Calibration Issues** (Mics are slightly off from their coordinates in `array_geo.txt`).

When this mismatch happens, standard LCMV assumes the slightly out-of-phase target is actually an *interferer* and places a deep spatial null on it—resulting in silence.

**How this simulator addresses it:**
We provide `RobustLCMV` variants:
- **Phase-STOR:** Normalizes amplitude spikes so loud transient noise (like wind) doesn't trick the interference subspace.
- **Covariance Matrix Tapering (CMT):** Mathematically widens the target acceptance beam, telling LCMV not to place nulls within a few degrees of the target, guaranteeing the target survives even with pointing errors.

---

## 2. Using Custom Array Geometries
The simulator natively supports any custom planar array layout via an `array_geo.txt` file in the root directory.

Format your file exactly like this:
```
x=[-0.15, -0.05, 0.05, 0.15, -0.15, -0.05, 0.05, 0.15]
y=[-0.15, -0.15, -0.15, -0.15, -0.05, -0.05, -0.05, -0.05]
```
If this file is present, the benchmarking script and the real audio processor will automatically parse and use it.

---

## 3. How to Run the Benchmark
The core benchmark orchestrator (`run_benchmark.py`) sweeps complex scenarios (like passing airplanes, look-errors, and multiple interferers) to evaluate the beamformers.

**To execute:**
```bash
# Set PYTHONPATH so the `simulator` module resolves correctly
PYTHONPATH=. python3 simulator/run_benchmark.py
```
**Outputs:**
1. A summary CSV of SI-SDR, LSD, and LAM performance metrics is written to `docs/benchmark_results.csv`.
2. Processed `.wav` files of the different beamformer outputs are written to `docs/audio_outputs/` so you can listen to the artifacts and self-cancellation yourself.

---

## 4. How to Process Real Multi-Channel Audio
If you have your own real multi-channel recordings (e.g., sliced into 2-second overlapping chunks), use the `process_real_audio.py` script.

**To execute:**
```bash
PYTHONPATH=. python3 process_real_audio.py
```

**How it works:**
1. Place your chunked `.wav` files into the `data/real_multichannel_chunks` directory.
2. The script processes each chunk individually using the **Wideband LCMV**.
3. It uses a **Linear Crossfade Overlap-Add (OLA)** algorithm across the 1-second overlap. This guarantees that amplitude exactly sums to `1.0` in the crossfade region, preventing any "pumping" or click artifacts in the final stitched audio track.
4. The stitched output is saved to `final_beamformed_output.wav`.

*(Note: If the `data/` folder is empty, the script will automatically generate synthetic physical multi-channel data—a 1000 Hz tone + noise—to demonstrate the pipeline).*

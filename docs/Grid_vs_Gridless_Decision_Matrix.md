# Decision Matrix: Grid Search vs. Gridless BSS Methods

To assist stakeholders in selecting the correct spatial audio architecture for embedded hardware deployment, we explicitly compare traditional Grid Search methods (like LCMV) against Gridless Blind Source Separation (like AuxIVA/ILRMA) based on our comprehensive empirical benchmarks.

## Executive Summary
*   **Choose Grid Search (SVE-LCMV)** if the system must handle highly dynamic moving targets, dense interference (many vehicles), and operates on extremely constrained RAM (microcontrollers).
*   **Choose Gridless (AuxIVA)** if the hardware supports moderate RAM, the target is relatively stationary within processing blocks, and you require the absolute maximum acoustic filtering quality without array calibration.

## Detailed Engineering Trade-Offs

| Feature / Constraint | Grid Search (SVE-LCMV) | Gridless BSS (AuxIVA / ILRMA) | Winner |
|:---|:---|:---|:---:|
| **Array Calibration Dependency** | **Moderate.** SVE can correct minor mismatch, but still relies on general array physics. | **None.** Completely blind to mic placement and amplitude drift. | **Gridless** |
| **Execution Speed (RTF)** | **Moderate to Slow.** Depends entirely on grid resolution (e.g., 64 vs 128 points). | **Fast (AuxIVA).** Separates mathematically in a single block pass without a spatial grid loop. | **Gridless (AuxIVA)** |
| **Peak Memory (RAM)** | **Low.** Can operate purely frame-by-frame with tiny memory footprints. | **High.** Requires retaining massive $F \times N \times M$ demixing matrices across the entire audio block. | **Grid** |
| **Target Tracking (Fast Motion)** | **Excellent.** Can dynamically adapt beamformer weights frame-by-frame to follow a fast drone. | **Poor.** Standard BSS assumes the target is stationary across the 1-2 second processing block. Fast motion blurs the separation. | **Grid** |
| **Multiple Interferer Capacity** | **High.** An $M$-microphone array can project $M-2$ deep nulls without significant penalty. | **Limited.** BSS must explicitly allocate an output channel for *every* distinct source ($N$). Increasing $N$ geometrically increases compute time. | **Grid** |
| **Target Ambiguity** | **None.** The algorithm points exactly at the physical coordinate requested. | **High.** BSS separates the audio into arbitrary tracks. A secondary classifier (e.g., LAM/YAMNet) must decide which track is the drone and which is the wind. | **Grid** |

## QC & Validation Data
All claims above are empirically backed by the synthetic and real-world benchmark datasets located in `docs/benchmark_gridless.csv` and validated by `experiments/qc_validation.py`. Under dense interference testing (1 Target + 3 Interferers), Gridless BSS (ILRMA) execution time doubled to ~13 seconds, whereas Grid LCMV execution time remained static at ~4.8 seconds, proving the Capacity limitations noted above.

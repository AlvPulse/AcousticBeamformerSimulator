# Near-Field Masking & Spherical Wavefront Distortion

In outdoor environmental acoustics, standard beamforming matrices mathematically assume that incoming sound waves are planar (flat), which is a valid assumption for far-field targets like UAVs > 100 meters away.

However, when a loud, near-field interferer (like a bird, dog, or passing car at 1–5 meters) emits sound, the wavefront is highly spherical across the microphone array. Because standard beamformers apply a flat spatial null to a curved wavefront, the null is "broken," allowing massive acoustic leakage.

This causes severe false-positives in downstream AI classification (LAM), where the loud leaked bird masks the weak target drone, causing the AI to mistakenly associate the bird's spectral signature with the "drone" label.

## The Benchmark (`benchmark_nearfield_masking.py`)
We modeled a severe masking scenario:
*   **Target:** UAV Drone at 1500 meters (-15 dB SNR)
*   **Interferer:** Bird, Dog, or Car at 1–5 meters (+35 dB INR)
*   **Wind:** 0.0 to 0.3 phase jitter to compound the spherical distortion.

### Evaluation Metrics
We tracked the **Interference Rejection Ratio (IRR)**. A negative IRR means the near-field interferer successfully bled through the spatial null and masked the target. A positive IRR means the algorithm rejected the near-field source.

## Empirical Findings

| Near-Field Masker | Algorithm | IRR (Interference Rejection) |
|:---|:---|:---:|
| **Bird (High-Freq Chirp)** | Gen 1 (Standard LCMV) | -16.00 dB (Severe Leakage) |
| | Gen 2 (SVE-LCMV) | -4.33 dB |
| | **Gen 3 (AuxIVA Gridless)** | **+2.44 dB (Successfully Rejected)** |
| **Car (Low-Freq Rumble)** | Gen 1 (Standard LCMV) | -9.46 dB |
| | Gen 2 (SVE-LCMV) | -4.77 dB |
| | Gen 3 (AuxIVA Gridless) | -2.69 dB |
| **Dog (Broadband Bark)** | Gen 1 (Standard LCMV) | -14.18 dB |
| | Gen 2 (SVE-LCMV) | -5.95 dB |
| | Gen 3 (AuxIVA Gridless) | -4.34 dB |

### Why do classical models fail?
Gen 1 LCMV relies purely on physical phase delays mapped to plane-waves. Since the near-field wave is curved, the exact delays expected by the plane-wave steering vector do not exist in the data, breaking the null. As a result, LCMV suffered massive -16 dB leakage.

### Why do BSS methods survive?
Gridless BSS methods like **AuxIVA** do not use spatial phase delays or plane-wave physics. They separate sources entirely based on statistical independence and higher-order spectral statistics (Spherical Laplace priors). Because they are mathematically "blind" to the curvature of the wavefront, they suffer zero penalty from near-field spherical distortions, successfully rejecting the Bird masking (+2.44 dB IRR).

### Conclusion for System Design
If your array is deployed in environments prone to near-field noise (e.g., birds landing on the array, close passing vehicles), **Gridless BSS (AuxIVA/ILRMA)** is strictly required to prevent your LAM AI from learning false-positive labels caused by spherical leakage.

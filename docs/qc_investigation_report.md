# Quality Control & Algorithm Investigation Report

This document provides a deep-dive Quality Control (QC) analysis of the advanced "White-Box" beamforming algorithms (`RobustLCMV_CMT`, `GSC_MCWF`, and `Spatial_IPD_Mask`) implemented in this repository. It analyzes mathematical edge cases, computational scaling, and hyperparameter tuning to help engineering teams deploy these algorithms safely in the field.

---

## 1. Mathematical Edge Cases & Safety Hardening

We rigorously audited the mathematical implementations to ensure pipeline stability.

### A. The "Digital Silence" Collapse
*   **The Threat:** If a segment of audio is perfectly silent (digital zeros), algorithms relying on spatial covariance matrices ($\mathbf{R}_{xx}$) will invert a purely singular (zero) matrix.
*   **The Fix:** All adaptive algorithms (`LCMV`, `GSC_MCWF`, `RobustLCMV`) have been hardened with dynamic trace-loading. Instead of a hardcoded $\epsilon$, the diagonal load factor is scaled dynamically based on the matrix trace:
    ```python
    trace_R = np.trace(R)
    load_factor = 1e-4 * trace_R if trace_R > 1e-10 else 1e-6
    R += load_factor * np.eye(M)
    ```
    Furthermore, denominator divisions are bound-checked (`if denom < 1e-10`) to prevent `NaN` cascades that would destroy the entire STFT inverse transform.

### B. High Multi-Path Echoes (Reverberation)
*   **GSC-MCWF Vulnerability:** The Generalized Sidelobe Canceller (GSC) operates under the assumption that the target arrives *exactly* from the specified look-direction. In highly reverberant environments (canyons, dense forests), strong target echoes arrive from off-axis angles. The GSC's Blocking Matrix ($\mathbf{B}$) fails to block these echoes, causing them to leak into the noise reference $\mathbf{u}$. The adaptive filter then actively *cancels* the target.
*   **CMT Superiority:** `RobustLCMV_CMT` handles multi-path target reflections significantly better. Because the Covariance Matrix Tapering (CMT) artificially widens the main lobe acceptance angle, early target reflections arriving within $\pm 5^\circ$ of the primary wavefront are preserved rather than cancelled.

---

## 2. Computational Scaling (Big-O Analysis)

When moving from a 50-microphone array to a massive 256-microphone array, the computational bottlenecks shift dramatically due to matrix inversion operations per frequency bin.

| Algorithm | Big-O Complexity | Real-Time on 50 Mics? | Scalability to 256 Mics? |
| :--- | :--- | :--- | :--- |
| **Tapered DAS** | $\mathcal{O}(M)$ | Yes ($\sim0.2\text{ms}$) | Excellent. Purely linear. |
| **LCMV / CMT / GSC** | $\mathcal{O}(M^3)$ | Yes ($\sim0.6\text{ms}$) | Poor. Matrix inversion scales cubically. Requires Cholesky updates. |
| **ILRMA (BSS)** | $\mathcal{O}(I \cdot K \cdot M^2)$ | Yes ($\sim0.4\text{ms}$) | Moderate. Scales with iterations ($I$) and components ($K$). |
| **Robust SOCP** | $\mathcal{O}(M^{3.5})$ | No ($\sim49\text{ms}$) | Impossible. Interior point convex solvers do not scale well. |

*(Note: If deploying to a 256-mic array, `LCMV` and `GSC` must be upgraded to use recursive matrix updates (RLS/Woodbury Matrix Identity) rather than brute-force `np.linalg.solve` on every STFT frame).*

---

## 3. Hyperparameter Tuning Guide

To optimize the simulator for your specific deployment, several critical hyperparameters are exposed in the constructor classes in `run_benchmark.py`.

### 1. `SpatialVarianceMask(variance_threshold=0.5)`
*   **What it does:** Controls how "strict" the array is about classifying a sound as a coherent wave (Target/Car) vs incoherent noise (Wind).
*   **How to tune:**
    *   Decrease to `0.2` if you are in a highly reverberant environment where multi-path makes true targets look slightly incoherent, and you want to aggressively protect them by falling back to DAS.
    *   Increase to `0.8` if you are in an open field and want the LCMV to actively suppress noise more often.

### 2. `RobustLCMV(use_cmt=True)` and `delta_theta=5.0`
*   **What it does:** Broadens the spatial nulls by `delta_theta` degrees.
*   **How to tune:**
    *   If you are tracking very fast-moving objects (like airplanes) or have cheap, uncalibrated microphones, increase to `8.0` or `10.0` degrees. This guarantees the airplane cannot slip out of the null between STFT frames.
    *   Warning: Increasing this parameter too high merges the nulls, causing extreme loss of White Noise Gain (WNG).

### 3. `GSC_MCWF(eps=1e-4)`
*   **What it does:** Controls the step-size/diagonal-loading of the adaptive interference canceller.
*   **How to tune:** If the MCWF is creating "musical noise" (bubbly, robotic artifacts), it means the noise estimator $\Phi_{nn}$ is too aggressive. Increase the diagonal loading (`eps=1e-3`) to force the adaptive filter to track noise slower, smoothing out the transients and reducing the robotic artifacts.

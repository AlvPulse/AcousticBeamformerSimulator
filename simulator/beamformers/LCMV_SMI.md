# Linearly Constrained Minimum Variance (LCMV) via SMI

## 1. The Mathematical Theory
The LCMV algorithm (often referred to as MVDR when constrained to a single point) is an adaptive beamformer that suppresses spatial interference. Unlike Delay-and-Sum (DAS) which has a fixed spatial response, LCMV dynamically computes its weights based on the actual acoustic environment.

It minimizes the total output power $\mathbf{w}^H \mathbf{R}_{xx} \mathbf{w}$ while ensuring unity gain in the exact look direction $\mathbf{a}_0$:
$$ \min_{\mathbf{w}} \mathbf{w}^H \mathbf{R}_{xx} \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^H \mathbf{a}_0 = 1 $$

The analytical solution, solved via Sample Matrix Inversion (SMI), is:
$$ \mathbf{w} = \frac{\mathbf{R}_{xx}^{-1} \mathbf{a}_0}{\mathbf{a}_0^H \mathbf{R}_{xx}^{-1} \mathbf{a}_0} $$

## 2. Implementation Details
In `lcmv_smi.py`:
1.  We compute the STFT of the multi-channel input.
2.  For every frequency bin $f$, we estimate the spatial covariance matrix $\mathbf{R}_{xx}(f) = \frac{1}{T} \mathbf{X} \mathbf{X}^H$.
3.  **Safety Guard:** To prevent the matrix from becoming singular during absolute silence, we apply dynamic diagonal loading: $\mathbf{R}_{load} = \mathbf{R}_{xx} + \epsilon \cdot \text{Tr}(\mathbf{R}_{xx}) \mathbf{I}$.
4.  We solve the linear equation $\mathbf{R}_{load} \mathbf{w} = \mathbf{a}_0$.

## 3. Benchmark Limitations (Target Self-Cancellation)
While LCMV is the optimal mathematical solution for stationary noise, it fails drastically outdoors. If the true target signal arrives even $1^\circ$ off from $\mathbf{a}_0$ (due to wind, movement, or calibration error), LCMV treats the target as interference and places a spatial null on it. In our benchmark, this "Target Self-Cancellation" caused the LAM Trust scores to drop to ~0.20 in windy scenarios.

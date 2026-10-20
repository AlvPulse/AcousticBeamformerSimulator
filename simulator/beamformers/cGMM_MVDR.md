# Mask-Based (cGMM) MVDR Beamformer

## 1. The Mathematical Theory
Standard LCMV (MVDR) computes a single covariance matrix $\mathbf{R}_{xx}$ that contains both the target signal and the noise. Because the target is present in the covariance matrix, any look-direction error causes the algorithm to cancel the target.

The **Mask-Based MVDR** solves this by estimating two entirely separate covariance matrices:
1.  **$\mathbf{R}_{ss}$ (Target Covariance):** Built only using STFT frames where the target is active.
2.  **$\mathbf{R}_{nn}$ (Noise Covariance):** Built only using STFT frames where the target is silent.

The optimal weights are then calculated using the Generalized Eigenvector of $(\mathbf{R}_{ss}, \mathbf{R}_{nn})$ or the standard MVDR formulation using the principal eigenvector of $\mathbf{R}_{ss}$ as the steering vector:
$$ \mathbf{w} = \frac{\mathbf{R}_{nn}^{-1} \mathbf{a}_{target}}{\mathbf{a}_{target}^H \mathbf{R}_{nn}^{-1} \mathbf{a}_{target}} $$

## 2. Implementation details
In actual deployments, the masks (determining which frame is "Target" and which is "Noise") are generated using complex Gaussian Mixture Models (cGMM) clustering the spatial features. In our `cgmm_mvdr.py` benchmark implementation, we use an energy-based Voice Activity Detection (VAD) proxy mask to establish the absolute upper-bound performance limit of this methodology.

## 3. Benchmark Performance
Because the target is explicitly removed from $\mathbf{R}_{nn}$, Mask-Based MVDR is highly robust against Target Self-Cancellation. However, in continuous environmental monitoring (like Airplanes or Drones), the target never "stops" making noise, making it incredibly difficult to obtain a pure $\mathbf{R}_{nn}$ matrix.

# Eigenspace-Based Beamformer (EBB)

## 1. The Mathematical Theory
The Eigenspace-Based Beamformer (EBB) is designed to explicitly combat the Target Self-Cancellation problem found in standard LCMV. It assumes that the true target steering vector is distorted, but still lies within a specific "Signal-plus-Interference" subspace of the spatial covariance matrix $\mathbf{R}_{xx}$.

By performing an Eigendecomposition on $\mathbf{R}_{xx}$, we can extract the eigenvectors corresponding to the largest $S$ eigenvalues (representing the dominant sound sources). This forms the subspace matrix $\mathbf{U}_s$.

We then mathematically project our "flawed" nominal steering vector $\mathbf{a}_0$ onto this subspace to find the "true" steering vector $\mathbf{\hat{a}}$:
$$ \mathbf{\hat{a}} = \mathbf{U}_s \mathbf{U}_s^H \mathbf{a}_0 $$

We then plug this updated $\mathbf{\hat{a}}$ back into the standard LCMV equation:
$$ \mathbf{w} = \frac{\mathbf{R}_{xx}^{-1} \mathbf{\hat{a}}}{\mathbf{\hat{a}}^H \mathbf{R}_{xx}^{-1} \mathbf{\hat{a}}} $$

## 2. Benchmark Performance
EBB works brilliantly in stable, high-SNR environments where the number of sources is strictly known. However, in our **Harsh Wind (S10)** scenario, EBB suffered a catastrophic collapse (Delta SI-SDR: -28 dB).
Because transient wind gusts hit the array randomly, the wind energy overpowers the target energy. The EBB algorithm accidentally extracts the *wind* into its subspace $\mathbf{U}_s$, projects the steering vector onto the wind, and completely cancels the target.

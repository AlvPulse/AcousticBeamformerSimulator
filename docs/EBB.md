# Eigenspace-Based Beamformer (EBB)

## Mathematical Overview
EBB improves upon LCMV by projecting the nominal steering vector onto the signal-plus-interference subspace. This prevents the beamformer from trying to null the target due to minor steering vector mismatch.

1. Eigendecomposition of $\mathbf{R}$: $\mathbf{R} = \mathbf{U}_s \mathbf{\Lambda}_s \mathbf{U}_s^H + \mathbf{U}_n \mathbf{\Lambda}_n \mathbf{U}_n^H$
2. Project nominal steering vector: $\hat{\mathbf{a}} = \mathbf{U}_s \mathbf{U}_s^H \mathbf{a}_0$
3. Apply MVDR with the projected vector:
$$\mathbf{w}_{EBB} = \frac{\mathbf{R}^{-1}\hat{\mathbf{a}}}{\hat{\mathbf{a}}^H \mathbf{R}^{-1}\hat{\mathbf{a}}}$$

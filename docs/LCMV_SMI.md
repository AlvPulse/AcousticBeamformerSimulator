# Linearly Constrained Minimum Variance (LCMV) & Sample Matrix Inversion (SMI)

## Mathematical Overview
LCMV is an adaptive beamformer that minimizes total output variance while ensuring a distortionless response in the target look direction.

Given a steering vector $\mathbf{a}(\theta, f)$ and a spatial covariance matrix $\mathbf{R}(f)$:
$$\min_{\mathbf{w}} \mathbf{w}^H \mathbf{R} \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^H \mathbf{a} = 1$$

Solution:
$$\mathbf{w}_{LCMV} = \frac{\mathbf{R}^{-1}\mathbf{a}}{\mathbf{a}^H \mathbf{R}^{-1}\mathbf{a}}$$

## Weaknesses in Real Data
LCMV is highly susceptible to **target self-cancellation** when the assumed steering vector $\mathbf{a}$ slightly mismatches the true acoustic wavefront (e.g., due to wind turbulence or calibration errors).

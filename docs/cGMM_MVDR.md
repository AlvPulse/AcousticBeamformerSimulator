# Complex Gaussian Mixture Model MVDR (cGMM-MVDR)

## Mathematical Overview
Instead of relying strictly on geometric steering vectors, cGMM-MVDR clusters time-frequency bins into "Target" and "Noise" spatial covariances using an Expectation-Maximization (EM) algorithm.

1. Estimate Target Covariance $\mathbf{\Phi}_X$ and Noise Covariance $\mathbf{\Phi}_N$.
2. Compute the Principal Eigenvector of $\mathbf{\Phi}_N^{-1}\mathbf{\Phi}_X$ to estimate the empirical Relative Transfer Function (RTF) $\mathbf{h}$.
3. Compute MVDR using the empirical RTF:
$$\mathbf{w}_{cGMM} = \frac{\mathbf{\Phi}_N^{-1}\mathbf{h}}{\mathbf{h}^H \mathbf{\Phi}_N^{-1}\mathbf{h}}$$

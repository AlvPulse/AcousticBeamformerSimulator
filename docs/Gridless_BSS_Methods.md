# Gridless Blind Source Separation (BSS)

In field acoustic testing, finding an unknown target conventionally requires steering a beamformer (like LCMV or MVDR) across a dense spatial grid (e.g., $N=64$ or $N=128$ azimuth/elevation coordinates) and measuring the output power at each point to identify the loudest source. This "Grid Search" approach multiplies the computational complexity by $N$, making it extremely slow.

To bypass grid-solving, we use **Gridless Methods**.

## 1. Auxiliary-function Independent Vector Analysis (AuxIVA)
**Concept:** AuxIVA is a SOTA gridless BSS algorithm. Unlike ILRMA which uses Non-Negative Matrix Factorization (NMF) to model complex source spectra, AuxIVA groups frequency bins belonging to the same source based on higher-order statistical dependencies (specifically, modeling the source with a super-Gaussian spherical Laplace prior).
**Advantage over Grid Search:** It separates $M$ microphones into $K$ independent audio sources in a single computational pass. It never computes a steering vector and never loops over spatial coordinates.
**Advantage over ILRMA:** It converges significantly faster than ILRMA because it does not have to simultaneously solve for an NMF basis matrix, drastically dropping the Real-Time Factor (RTF).

## 2. ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
**Concept:** ESPRIT belongs to the family of high-resolution subspace methods (like MUSIC). However, while MUSIC requires a grid search over all angles to find peaks in a pseudo-spectrum, ESPRIT solves for the Angles of Arrival (DoA) analytically using eigenvalues of the rotational shift between sub-arrays.
**Advantage:** Instantly gives the exact angle of arrival mathematically without searching a grid. Once the angle is known, a standard LCMV can be steered to it exactly once.

## 3. Statistical T-F Masking (Gridless cGMM)
**Concept:** Complex Gaussian Mixture Models cluster the raw Time-Frequency (STFT) bins directly into "Target" and "Noise" categories based on their spatial covariance, without knowing where those clusters physically exist in the world.

## Summary for Implementation
For our array benchmarking pipeline, testing an iterative grid-search LCMV against a one-shot AuxIVA/ILRMA separation mathematically proves the computational savings of gridless methods.

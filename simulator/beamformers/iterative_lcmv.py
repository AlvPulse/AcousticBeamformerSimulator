import numpy as np
import scipy.linalg as sp_linalg
from simulator.beamformers.base import BaseBeamformer
from simulator.beamformers.lcmv_smi import LCMVSMI

class IterativeLCMV(BaseBeamformer):
    """
    Iterative LCMV (Iterative Source Detection and Localization).
    Implements a greedy "Peak Depletion" mechanism modeled after IASDNet.
    Instead of assuming a single target or blindly separating N sources,
    this beamformer iteratively finds the dominant spatial peak, extracts the signal,
    projects that signal's covariance OUT of the data, and searches the residual spectrum
    for the next peak until the residual energy falls below a noise threshold.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, max_sources=4, energy_threshold_ratio=0.1):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.max_sources = max_sources
        self.energy_threshold_ratio = energy_threshold_ratio

    def process(self, audio_mc, look_pos=None, **kwargs):
        # We need a predefined spatial grid to search for peaks iteratively
        # In a real system, this could be a full 2D grid. For speed, we use 1D azimuth.
        azimuths = np.linspace(-180, 180, 64, endpoint=False)
        grid_vectors = [
            np.array([120.0 * np.cos(np.deg2rad(az)), 120.0 * np.sin(np.deg2rad(az)), 0.0])
            for az in azimuths
        ]

        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        # We will hold the residual covariance matrices here
        R_residual = np.zeros((F, M, M), dtype=np.complex128)
        for f in range(F):
            X_f = Zxx[:, f, :]
            R_residual[f] = (X_f @ np.conj(X_f).T) / T

        initial_trace = np.sum([np.trace(R_residual[f]) for f in range(F)])

        extracted_sources = []

        # Pre-compute steering vectors for the entire grid
        A_grid = np.zeros((len(grid_vectors), F, M), dtype=np.complex128)
        for idx, pos in enumerate(grid_vectors):
            A_grid[idx] = self.compute_steering_vector(pos)

        # Iterative Peak Depletion Loop
        for iter_idx in range(self.max_sources):
            # 1. Spatial Spectrum Search on the RESIDUAL covariance
            spatial_spectrum = np.zeros(len(grid_vectors))

            # Temporary weights for this iteration
            W_grid = np.zeros((len(grid_vectors), F, M), dtype=np.complex128)

            for f in range(F):
                R_load = R_residual[f] + 1e-3 * np.trace(R_residual[f]) * np.eye(M)
                try:
                    R_inv = np.linalg.inv(R_load)
                except np.linalg.LinAlgError:
                    R_inv = np.linalg.pinv(R_load)

                for idx in range(len(grid_vectors)):
                    a_f = A_grid[idx, f, :]
                    num = R_inv @ a_f
                    den = np.conj(a_f).T @ num
                    w = num / (den + 1e-12)
                    W_grid[idx, f, :] = w
                    # P(theta) = 1 / (a^H R^-1 a)
                    spatial_spectrum[idx] += np.real(1.0 / (den + 1e-12))

            # 2. Find the dominant peak in the residual spectrum
            best_idx = np.argmax(spatial_spectrum)

            # Beamform to extract the signal at this peak
            Y_f = np.zeros((F, T), dtype=np.complex128)
            best_W = W_grid[best_idx]
            for f in range(F):
                Y_f[f, :] = np.conj(best_W[f, :]).T @ Zxx[:, f, :]

            current_energy = np.sum(np.abs(Y_f)**2)

            # 3. Activity Threshold check (IASDNet equivalent 0/1 classifier)
            # If the energy of this extracted source is less than X% of the original trace,
            # we assume it's just background noise and break.
            if iter_idx > 0 and current_energy < (self.energy_threshold_ratio * np.real(initial_trace) * T):
                break

            extracted_sources.append(Y_f)

            # 4. Peak Depletion
            # Subtract the rank-1 covariance of the extracted signal from the residual covariance
            for f in range(F):
                # Steering vector of the extracted target
                a_f = A_grid[best_idx, f, :]
                # Target variance at this freq
                var_s = np.mean(np.abs(Y_f[f, :])**2)
                # Outer product
                R_target = var_s * np.outer(a_f, np.conj(a_f))
                # Deplete
                R_residual[f] = R_residual[f] - R_target

                # Ensure positive semi-definite (clip negative eigenvalues caused by over-subtraction)
                eigvals, eigvecs = np.linalg.eigh(R_residual[f])
                eigvals[eigvals < 0] = 1e-12
                R_residual[f] = eigvecs @ np.diag(eigvals) @ np.conj(eigvecs).T

        # If it found multiple targets, for the sake of the base API which returns (n_samples,),
        # we return the loudest one (index 0).
        # In a full tracking pipeline, we would return all `extracted_sources`.
        if len(extracted_sources) > 0:
            return self.istft(extracted_sources[0])
        else:
            return np.zeros(audio_mc.shape[1])

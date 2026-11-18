import numpy as np
from simulator.beamformers.base import BaseBeamformer

class RobustLCMV(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3, cmt_width=0.05):
        """
        Robust LCMV using Covariance Matrix Tapering (CMT) for null broadening
        and robustness against steering vector mismatch.

        cmt_width is the rms angular spread (radians) the nulls should cover.
        The taper is frequency-dependent: T_ij(f) = sinc(2 f sigma_theta d_ij / c).
        (The previous constant taper sinc(0.05 * d_ij) evaluated to ~identity
        for a 2.5 m array -- i.e. no null broadening at all.)
        """
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon
        self.cmt_width = cmt_width

        M = self.n_mics
        dist_mat = np.linalg.norm(pos_mics[:, None, :] - pos_mics[None, :, :], axis=2)

        # Precompute taper per frequency bin: (F, M, M)
        F = len(self.freqs)
        self.T_cmt = np.ones((F, M, M))
        for f_idx, f in enumerate(self.freqs):
            arg = 2.0 * f * self.cmt_width * dist_mat / self.c
            self.T_cmt[f_idx] = np.sinc(arg / np.pi)  # np.sinc(x) = sin(pi x)/(pi x)

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]

            # Sample Covariance Matrix
            R = (X_f @ np.conj(X_f).T) / T

            # Apply Covariance Matrix Tapering (CMT)
            R_cmt = R * self.T_cmt[f]

            # Diagonal loading
            R_load = R_cmt + self.epsilon * np.trace(R_cmt) * np.eye(M)

            # Stable eigendecomposition-based inverse with eigenvalue floor
            # (np.linalg.pinv can stall on pathological covariances).
            eigvals, eigvecs = np.linalg.eigh(R_load)
            floor = self.epsilon * max(float(np.max(eigvals)), 1e-30)
            eigvals = np.maximum(eigvals, floor)
            R_inv = (eigvecs * (1.0 / eigvals)) @ np.conj(eigvecs).T

            a_f = a[f, :]

            numerator = R_inv @ a_f
            denominator = np.conj(a_f).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_f / M
            else:
                w = numerator / denominator

            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

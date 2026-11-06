import numpy as np
import scipy.linalg as sp_linalg
from simulator.beamformers.base import BaseBeamformer

class SVE_LCMV(BaseBeamformer):
    """
    Steering Vector Estimation (SVE) LCMV.
    Instead of relying purely on the mathematical steering vector from the physical geometry,
    which collapses in near-field/multipath conditions, this estimates the TRUE steering vector
    from the data covariance matrix (using the principal eigenvector during high-energy frames).
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        # Initial theoretical vector (used to resolve phase ambiguity)
        a_nom = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        # Estimate Target presence (Simple VAD based on energy)
        frame_energies = np.sum(np.abs(Zxx)**2, axis=(0, 1))
        p_target = np.percentile(frame_energies, 80)
        target_mask = frame_energies > p_target

        if np.sum(target_mask) < 2:
            target_mask = np.ones(T, dtype=bool)

        for f in range(F):
            X_f = Zxx[:, f, :]

            # Target Covariance Matrix
            X_target = X_f[:, target_mask]
            R_target = (X_target @ np.conj(X_target).T) / max(1, X_target.shape[1])

            # Full Data Covariance Matrix
            R = (X_f @ np.conj(X_f).T) / T

            # Extract true steering vector: Principal eigenvector of R_target
            eigvals, eigvecs = np.linalg.eigh(R_target)
            a_est = eigvecs[:, -1]

            # Resolve phase ambiguity using nominal steering vector
            # We want a_est to point generally in the same direction as a_nom
            phase_align = np.exp(-1j * np.angle(np.conj(a_nom[f, :]).T @ a_est))
            a_est = a_est * phase_align

            # Normalize
            a_est = a_est / (np.linalg.norm(a_est) + 1e-12) * np.sqrt(M)

            # Apply Diagonal Loading
            R_load = R + self.epsilon * np.trace(R) * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_load)

            # LCMV Weights using ESTIMATED steering vector
            numerator = R_inv @ a_est
            denominator = np.conj(a_est).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_est / M
            else:
                w = numerator / denominator

            Y[f, :] = np.conj(w).T @ X_f

        y_out = self.istft(Y)
        return y_out

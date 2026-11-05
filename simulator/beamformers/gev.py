import numpy as np
import scipy.linalg as sp_linalg
from simulator.beamformers.base import BaseBeamformer

class GEV(BaseBeamformer):
    """
    Generalized Eigenvalue (GEV) Beamformer (Blind Max-SNR).
    Maximizes SNR by finding the principal generalized eigenvector of the
    target covariance matrix and noise covariance matrix.
    Since it is data-driven, it is highly robust to uncalibrated arrays.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256):
        super().__init__(pos_mics, fs, n_fft, hop_length)

    def process(self, audio_mc, look_pos=None, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        Y = np.zeros((F, T), dtype=np.complex128)

        # In a real GEV, you need a VAD mask to estimate target (R_x) and noise (R_n).
        # For this blind offline simulation, we approximate R_x from high-energy frames
        # and R_n from low-energy frames (simple energy-based VAD).

        # Calculate broadband energy per frame
        frame_energies = np.sum(np.abs(Zxx)**2, axis=(0, 1))

        # Use top 30% of frames as target, bottom 30% as noise
        p70 = np.percentile(frame_energies, 70)
        p30 = np.percentile(frame_energies, 30)

        mask_x = frame_energies > p70
        mask_n = frame_energies < p30

        # If signal is entirely stationary (masks fail), fallback to all frames
        if np.sum(mask_x) == 0 or np.sum(mask_n) == 0:
            mask_x = np.ones(T, dtype=bool)
            mask_n = np.ones(T, dtype=bool)

        for f in range(F):
            X_f = Zxx[:, f, :]

            # Target Covariance
            X_target = X_f[:, mask_x]
            R_x = (X_target @ np.conj(X_target).T) / max(1, X_target.shape[1])

            # Noise Covariance
            X_noise = X_f[:, mask_n]
            R_n = (X_noise @ np.conj(X_noise).T) / max(1, X_noise.shape[1])

            # Regularize noise matrix
            R_n += 1e-4 * np.trace(R_n) * np.eye(M)

            # Solve Generalized Eigenvalue Problem: R_x w = lambda R_n w
            try:
                # eigh works since R_x and R_n are Hermitian
                eigvals, eigvecs = sp_linalg.eigh(R_x, R_n)
                # Max SNR is the eigenvector corresponding to largest eigenvalue
                w = eigvecs[:, -1]
            except np.linalg.LinAlgError:
                # Fallback to standard principal component if GEV fails
                eigvals, eigvecs = np.linalg.eigh(R_x)
                w = eigvecs[:, -1]

            # Beamform
            Y[f, :] = np.conj(w).T @ X_f

        y_out = self.istft(Y)
        return y_out

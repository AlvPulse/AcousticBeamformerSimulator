import numpy as np
from simulator.beamformers.base import BaseBeamformer

class GSC_MCWF(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, mu=0.01):
        """
        Generalized Sidelobe Canceller with Multi-Channel Wiener Filter.
        Uses a blocking matrix to isolate noise, then applies MCWF.
        """
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.mu = mu

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :] # (M, T)
            a_f = a[f, :]      # (M,)

            # Fixed Beamformer (FBF) branch: standard Delay-and-Sum
            y_fbf = np.conj(a_f).T @ X_f # (T,)

            # Blocking Matrix (B): Must be orthogonal to a_f
            # B^H a_f = 0
            # We can construct B using Gram-Schmidt or projection
            P_perp = np.eye(M) - np.outer(a_f, np.conj(a_f)) / (np.linalg.norm(a_f)**2 + 1e-12)

            # Get full orthogonal basis, keep M-1 vectors
            U, S, Vh = np.linalg.svd(P_perp)
            B = U[:, :M-1] # (M, M-1)

            # Noise reference signals
            U_noise = np.conj(B).T @ X_f # (M-1, T)

            # MCWF / Adaptive filter to estimate y_fbf from U_noise
            # We want to minimize E[|y_fbf - W^H U_noise|^2]
            # W = R_uu^-1 r_uy

            R_uu = (U_noise @ np.conj(U_noise).T) / T
            r_uy = (U_noise @ np.conj(y_fbf).T) / T

            # Diagonal loading for robustness
            R_uu += 1e-3 * np.trace(R_uu) * np.eye(M-1)

            try:
                W = np.linalg.inv(R_uu) @ r_uy
            except np.linalg.LinAlgError:
                W = np.linalg.pinv(R_uu) @ r_uy

            # Cancel noise
            y_cancel = np.conj(W).T @ U_noise
            Y[f, :] = y_fbf - y_cancel

        return self.istft(Y)

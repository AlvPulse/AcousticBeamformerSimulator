import numpy as np
from simulator.beamformers.base import BaseBeamformer

class LCMVSMI(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)

        Y = np.zeros((F, T), dtype=np.complex128)

        # Process frequency by frequency
        for f in range(F):
            # X_f: (M, T)
            X_f = Zxx[:, f, :]

            # Covariance matrix R: (M, M)
            R = (X_f @ np.conj(X_f).T) / T

            # Diagonal loading
            R_load = R + self.epsilon * np.trace(R) * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                R_inv = np.linalg.pinv(R_load)

            # Steering vector for this freq: (M,)
            a_f = a[f, :]

            # MVDR / LCMV (rank 1 constraint) weight
            numerator = R_inv @ a_f
            denominator = np.conj(a_f).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_f / M
            else:
                w = numerator / denominator

            # Beamform
            Y[f, :] = np.conj(w).T @ X_f

        y_out = self.istft(Y)
        return y_out

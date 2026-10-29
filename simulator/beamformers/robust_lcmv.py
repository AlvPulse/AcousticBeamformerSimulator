import numpy as np
from simulator.beamformers.base import BaseBeamformer

class RobustLCMV(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3, cmt_width=0.05):
        """
        Robust LCMV using Covariance Matrix Tapering (CMT) for null broadening
        and robustness against steering vector mismatch.
        """
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon
        self.cmt_width = cmt_width

        # Precompute CMT taper matrix (spatial sinc function based on distance)
        M = self.n_mics
        self.T_cmt = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                dist = np.linalg.norm(pos_mics[i] - pos_mics[j])
                # Simple taper: sinc(alpha * distance)
                # alpha determines the width of the broadened nulls
                arg = self.cmt_width * dist
                if arg == 0:
                    self.T_cmt[i, j] = 1.0
                else:
                    self.T_cmt[i, j] = np.sin(arg) / arg

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
            R_cmt = R * self.T_cmt

            # Diagonal loading
            R_load = R_cmt + self.epsilon * np.trace(R_cmt) * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_load)

            a_f = a[f, :]

            numerator = R_inv @ a_f
            denominator = np.conj(a_f).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_f / M
            else:
                w = numerator / denominator

            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

import numpy as np
from simulator.beamformers.base import BaseBeamformer

class EBB(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256):
        super().__init__(pos_mics, fs, n_fft, hop_length)

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]
            R = (X_f @ np.conj(X_f).T) / T

            eigvals, eigvecs = np.linalg.eigh(R)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # Subspace dimension: roughly based on energy, or fixed rank.
            # Signal + Interference subspace. Let's assume energy threshold (e.g. 90%)
            energy_cumsum = np.cumsum(eigvals)
            threshold = 0.95 * energy_cumsum[-1]
            K = np.searchsorted(energy_cumsum, threshold) + 1
            K = min(K, M - 1)

            U_s = eigvecs[:, :K]

            # Project nominal steering vector onto signal+interference subspace
            a_hat = U_s @ np.conj(U_s).T @ a[f, :]

            # Diagonal loading for MVDR on reconstructed vector
            epsilon = 1e-3
            R_load = R + epsilon * np.trace(R) * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_load)

            numerator = R_inv @ a_hat
            denominator = np.conj(a_hat).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_hat / M
            else:
                w = numerator / denominator

            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

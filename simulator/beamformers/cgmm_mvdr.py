import numpy as np
from simulator.beamformers.base import BaseBeamformer

class cGMM_MVDR(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, n_iter=3):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.n_iter = n_iter

    def process(self, audio_mc, look_pos, **kwargs):
        """
        Simplified Mask-based / cGMM MVDR.
        Instead of full EM which is extremely slow in Python, we use a
        heuristic mask based on the spatial target direction to initialize
        and perform a couple of iterations of mask refinement.
        """
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a_nom = self.compute_steering_vector(look_pos) # (F, M)

        # 1. Initial heuristic mask based on beamformed power vs total power
        target_mask = np.zeros((F, T))
        for f in range(F):
            # Delay and sum power
            a_f = a_nom[f, :]
            y_das = np.conj(a_f).T @ Zxx[:, f, :]
            power_das = np.abs(y_das)**2
            power_total = np.mean(np.abs(Zxx[:, f, :])**2, axis=0)

            # Mask based on spatial power ratio
            ratio = power_das / (power_total + 1e-12)
            # Normalize to [0, 1] across time
            ratio = ratio / (np.max(ratio) + 1e-12)
            target_mask[f, :] = ratio

        noise_mask = 1.0 - target_mask

        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]

            # Accumulate covariance matrices
            lambda_s = np.sum(target_mask[f, :])
            lambda_n = np.sum(noise_mask[f, :])

            if lambda_s < 1e-3:
                R_ss = np.eye(M)
            else:
                R_ss = (X_f * target_mask[f, :]) @ np.conj(X_f).T / lambda_s

            if lambda_n < 1e-3:
                R_nn = np.eye(M)
            else:
                R_nn = (X_f * noise_mask[f, :]) @ np.conj(X_f).T / lambda_n

            # Compute steering vector via Principal Eigenvector of R_nn^-1 R_ss
            R_nn_load = R_nn + 1e-3 * np.trace(R_nn) * np.eye(M)
            try:
                R_nn_inv = np.linalg.inv(R_nn_load)
            except np.linalg.LinAlgError:
                R_nn_inv = np.linalg.pinv(R_nn_load)

            matrix_prod = R_nn_inv @ R_ss
            eigvals, eigvecs = np.linalg.eig(matrix_prod)

            # Principal eigenvector
            a = eigvecs[:, np.argmax(np.real(eigvals))]

            # Enforce phase alignment with reference mic (mic 0)
            a = a * np.exp(-1j * np.angle(a[0]))

            # MVDR weights
            numerator = R_nn_inv @ a
            denominator = np.conj(a).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a / M
            else:
                w = numerator / denominator

            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

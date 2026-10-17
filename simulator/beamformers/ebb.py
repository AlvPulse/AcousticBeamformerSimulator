import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class EBB(BaseBeamformer):
    """
    Eigenspace-Based Beamformer (EBB)
    Projects the nominal steering vector onto the signal-plus-interference subspace
    to create an updated steering vector before solving the standard MVDR/LCMV problem.
    """
    def __init__(self, rank_s=2, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.rank_s = rank_s
        self.eps = eps

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        Y = np.zeros((F, T), dtype=complex)

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :]
            R = (X_f @ np.conj(X_f.T)) / T
            R_load = R + self.eps * np.trace(R) * np.eye(M) / M

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(R)

            # Sort descending
            sort_eigs = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_eigs]

            # Extract signal+interference subspace U_s
            U_s = eigvecs[:, :self.rank_s]

            # Nominal steering vector
            a_0 = self.compute_steering_vector(array_positions, f, target_az, target_el)

            # Projected steering vector
            a_hat = (U_s @ np.conj(U_s.T)) @ a_0

            try:
                R_inv_a = np.linalg.solve(R_load, a_hat)
                denominator = np.conj(a_hat) @ R_inv_a
                w = R_inv_a / np.real(denominator)
            except np.linalg.LinAlgError:
                w = a_hat / M

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

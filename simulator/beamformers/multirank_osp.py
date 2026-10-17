import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class MultiRankOSP(BaseBeamformer):
    """
    Multi-Rank Orthogonal Subspace Projection (OSP)
    Extracts the interference subspace U_I using Eigendecomposition and
    orthogonally projects the nominal steering vector away from it.
    """
    def __init__(self, rank_k=1, **kwargs):
        super().__init__(**kwargs)
        self.rank_k = rank_k

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        Y = np.zeros((F, T), dtype=complex)

        # Nominal spatial taper (Chebyshev proxy)
        taper_1d = scipy.signal.windows.chebwin(M, at=40)
        radii = np.linalg.norm(array_positions, axis=1)
        sort_idx = np.argsort(radii)
        w_cheby = np.zeros(M)
        w_cheby[sort_idx] = taper_1d
        w_cheby /= np.sum(w_cheby)

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :]
            R = (X_f @ np.conj(X_f.T)) / T

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(R)

            # Sort descending
            sort_eigs = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_eigs]

            # Extract interference subspace U_I
            U_I = eigvecs[:, :self.rank_k]

            # Orthogonal projection matrix: P_I_perp = I - U_I * U_I^H
            P_I_perp = np.eye(M) - U_I @ np.conj(U_I.T)

            a_s = self.compute_steering_vector(array_positions, f, target_az, target_el)

            # Project Chebyshev weights
            w_proj = P_I_perp @ (w_cheby * a_s)
            denominator = np.conj(a_s) @ w_proj
            if np.abs(denominator) > 1e-10:
                w = w_proj / denominator
            else:
                w = w_cheby * a_s

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

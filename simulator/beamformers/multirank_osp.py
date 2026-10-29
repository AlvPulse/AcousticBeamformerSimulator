import numpy as np
import scipy.signal.windows as windows
from simulator.beamformers.base import BaseBeamformer

class MultiRankOSP(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, sidelobe_att_db=50):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.spatial_window = windows.chebwin(self.n_mics, at=sidelobe_att_db)
        self.spatial_window /= np.sum(self.spatial_window)

    def _estimate_rank_mdl(self, eigenvalues, T):
        """
        Estimates the number of interference sources using Minimum Description Length (MDL).
        """
        M = len(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1] # Descending
        mdl = np.zeros(M)

        for k in range(M):
            # Noise variance estimate
            if k == M - 1:
                sigma2 = eigenvalues[-1]
                num = eigenvalues[-1]
            else:
                sigma2 = np.mean(eigenvalues[k:])
                num = np.prod(eigenvalues[k:])**(1.0/(M-k))

            if sigma2 <= 0:
                L = 0
            else:
                ratio = num / sigma2
                if ratio <= 0:
                    L = 0
                else:
                    L = (M - k) * T * np.log(ratio)

            penalty = 0.5 * k * (2 * M - k) * np.log(T)
            mdl[k] = -L + penalty

        return np.argmin(mdl)

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]
            R = (X_f @ np.conj(X_f).T) / T

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(R)

            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # MDL to find K (interference rank)
            # We assume K > 0 and K < M, cap at max 10 interferers for stability
            K = self._estimate_rank_mdl(eigvals, T)
            K = min(K, M // 2)

            # Subspace projection
            w_cheby = a[f, :] * self.spatial_window

            if K > 0:
                U_I = eigvecs[:, :K]
                P_I_perp = np.eye(M) - U_I @ np.conj(U_I).T

                # Projected weight
                numerator = P_I_perp @ w_cheby
                denominator = np.conj(a[f, :]).T @ numerator

                if np.abs(denominator) < 1e-12:
                    w = w_cheby
                else:
                    w = numerator / denominator
            else:
                w = w_cheby

            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

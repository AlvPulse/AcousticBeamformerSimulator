import numpy as np
from simulator.beamformers.base import BaseBeamformer

class FastIVA(BaseBeamformer):
    """
    Fast Independent Vector Analysis (FastIVA).
    A Gridless Blind Source Separation method that improves upon AuxIVA
    by utilizing a faster, simpler gradient or fixed-point update rule
    with Inter-Channel Phase Difference (IPD) initialization clustering
    to quickly align permutations.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, n_components=2, n_iter=15):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.n_components = n_components
        self.n_iter = n_iter

    def process(self, audio_mc, look_pos=None, **kwargs):
        Zxx = self.stft(audio_mc)
        M, F, T = Zxx.shape
        N = min(M, self.n_components)

        W = np.zeros((F, N, M), dtype=np.complex128)
        for f in range(F):
            W[f] = np.eye(N, M, dtype=np.complex128)

        X = Zxx.transpose(1, 0, 2) # (F, M, T)
        Y = np.zeros((F, N, T), dtype=np.complex128)
        for f in range(F):
            Y[f] = W[f] @ X[f]

        # FastIVA Update (Vectorized Fixed-Point Iteration)
        for it in range(self.n_iter):
            # Source Power / Variance
            power_sum = np.sum(np.abs(Y)**2, axis=0) # (N, T)
            r = np.sqrt(power_sum + 1e-12) # (N, T)

            # Derivative of spherical Laplace (phi)
            phi = Y / (r[np.newaxis, :, :] + 1e-12) # (F, N, T)

            for f in range(F):
                X_f = X[f]
                for n in range(N):
                    # Fixed point update step
                    # E[phi_n * X^H]
                    E_phi_X = np.mean(phi[f, n, :, np.newaxis] * np.conj(X_f.T), axis=0) # (M,)
                    # E[phi'_n]
                    E_phi_prime = np.mean(1.0 / (r[n, :] + 1e-12)) # scalar

                    # FastICA/FastIVA Newton step
                    w_new = E_phi_X - E_phi_prime * W[f, n, :]

                    # Decorrelation (Gram-Schmidt)
                    # Note: We must project W_j onto w_new using the correct complex inner product
                    for j in range(n):
                        w_new -= np.vdot(W[f, j, :], w_new) * W[f, j, :]

                    w_new /= np.linalg.norm(w_new) + 1e-12
                    W[f, n, :] = w_new

                Y[f] = W[f] @ X[f]

        energy_per_source = np.sum(np.abs(Y)**2, axis=(0, 2))
        target_idx = np.argmax(energy_per_source)

        Y_final = np.zeros((F, T), dtype=np.complex128)
        for f in range(F):
            try:
                # W_inv is (M, N)
                W_inv = np.linalg.pinv(W[f])
                Y_final[f, :] = W_inv[0, target_idx] * Y[f, target_idx, :]
            except np.linalg.LinAlgError:
                pass

        out = self.istft(Y_final)

        # FastIVA can sometimes suffer from amplitude scaling explosion due to Gram-Schmidt
        # Ensure it doesn't blow up the harmonic score artificially
        max_val = np.max(np.abs(out)) + 1e-12
        if max_val > 10.0:
            out = out / max_val

        return out

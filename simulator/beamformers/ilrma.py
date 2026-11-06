import numpy as np
from simulator.beamformers.base import BaseBeamformer

class ILRMA(BaseBeamformer):
    """
    Independent Low-Rank Matrix Analysis (ILRMA).
    Blind Source Separation (BSS) technique that assumes sources are statistically
    independent and have low-rank spectrogram structures (NMF).
    Highly robust to array mismatch and near-field effects because it DOES NOT
    use the array geometry or steering vector.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, n_components=2, n_bases=2, n_iter=20):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.n_components = n_components # Number of sources to separate (e.g. target + noise)
        self.n_bases = n_bases # NMF bases per source
        self.n_iter = n_iter

    def process(self, audio_mc, look_pos=None, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        # We can only separate up to M sources.
        N = min(M, self.n_components)

        # Initialize Demixing Matrices W: (F, N, M)
        W = np.zeros((F, N, M), dtype=np.complex128)
        for f in range(F):
            W[f] = np.eye(N, M, dtype=np.complex128)

        # Initialize NMF variables
        # T_nf = Sum_k T_nkf V_nkt
        # Basis matrix T: (N, n_bases, F)
        T_mat = np.random.rand(N, self.n_bases, F) + 0.1
        # Activation matrix V: (N, n_bases, T)
        V_mat = np.random.rand(N, self.n_bases, T) + 0.1

        # Precompute input power
        X = Zxx.transpose(1, 0, 2) # (F, M, T)

        # ILRMA Iterations (Simplified IP update)
        for it in range(self.n_iter):
            # 1. Update source variances (R) from NMF
            # R: (N, F, T)
            R = np.zeros((N, F, T))
            for n in range(N):
                for f in range(F):
                    R[n, f, :] = np.sum(T_mat[n, :, f, np.newaxis] * V_mat[n, :, :], axis=0)

            # 2. Update Demixing Matrices W (Iterative Projection)
            for f in range(F):
                X_f = X[f]
                for n in range(N):
                    # Compute weighted covariance
                    V_inv = 1.0 / (R[n, f, :] + 1e-12) # (T,)
                    # weighted cov: (M, M)
                    U = (X_f * V_inv) @ np.conj(X_f).T / T

                    try:
                        # Fallback for overdetermined BSS where standard IP breaks
                        # Use a simple pseudo-inverse mapping: W_new approx (U * W^H)^-1
                        # Since standard IP expects M=N (square), and here N < M:

                        # Simplified update: Project onto subspace
                        # W[f] @ U is (N, M). pseudo inverse is (M, N).
                        # np.eye(N)[n] is (N,).
                        # pinv(W[f]@U) @ e_n is (M,).
                        w_unscaled = np.linalg.pinv(W[f] @ U) @ np.eye(N)[n]
                        # w_unscaled is now Mx1 (or (M,) array)
                        # We just want w_n_new to be w_unscaled (it already maps from the pseudo inverse)
                        w_n_new = w_unscaled

                        scale = np.sqrt(np.real(np.conj(w_n_new).T @ U @ w_n_new) + 1e-12)

                        W[f, n, :] = np.conj(w_n_new / scale).flatten()

                    except np.linalg.LinAlgError:
                        pass

            # 3. Separate signals with current W
            # Y: (F, N, T)
            Y = np.zeros((F, N, T), dtype=np.complex128)
            for f in range(F):
                Y[f] = W[f] @ X[f]

            # 4. Update NMF Parameters (T_mat, V_mat) using IS divergence rules
            P = np.abs(Y)**2 # (F, N, T)
            for n in range(N):
                P_n = P[:, n, :] # (F, T)
                R_n = R[n] # (F, T)

                # Update T_mat
                # T_mat is (N, n_bases, F), so T_mat[n] is (n_bases, F)
                # V_mat is (N, n_bases, T), so V_mat[n] is (n_bases, T)
                # (P_n / R_n^2) is (F, T).
                # (F, T) @ (T, n_bases) = (F, n_bases). Transpose to get (n_bases, F)
                num_T = ((P_n / (R_n**2 + 1e-12)) @ V_mat[n].T).T
                den_T = ((1.0 / (R_n + 1e-12)) @ V_mat[n].T).T
                T_mat[n] *= np.sqrt(num_T / (den_T + 1e-12))

                # Recompute R_n
                R_n = np.sum(T_mat[n, :, :, np.newaxis] * V_mat[n, :, np.newaxis, :], axis=0)

                # Update V_mat
                # (n_bases, F) @ (F, T) = (n_bases, T)
                num_V = T_mat[n] @ (P_n / (R_n**2 + 1e-12))
                den_V = T_mat[n] @ (1.0 / (R_n + 1e-12))
                V_mat[n] *= np.sqrt(num_V / (den_V + 1e-12))

        # After convergence, we have N separated sources.
        # BSS suffers from permutation ambiguity. We must select the target.
        # Since ILRMA separates sources, one will be the drone (tonal) and one will be noise.
        # We can pick the source with the highest tonality (kurtosis or peak variance).
        # For simplicity in this blind setup, we pick the one with max energy.
        Y_final = np.zeros((F, T), dtype=np.complex128)

        # We must recompute Y for the final time.
        for f in range(F):
            Y[f] = W[f] @ X[f]

        energy_per_source = np.sum(np.abs(Y)**2, axis=(0, 2))
        target_idx = np.argmax(energy_per_source)

        # Apply back-projection to fix scale ambiguity (Projection Back)
        for f in range(F):
            try:
                W_inv = np.linalg.pinv(W[f])
                # Back-project to microphone 0
                Y_final[f, :] = W_inv[0, target_idx] * Y[f, target_idx, :]
            except np.linalg.LinAlgError:
                pass

        return self.istft(Y_final)

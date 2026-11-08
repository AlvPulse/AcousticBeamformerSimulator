import numpy as np
from simulator.beamformers.base import BaseBeamformer

class AuxIVA(BaseBeamformer):
    """
    Auxiliary-function Independent Vector Analysis (AuxIVA).
    A fast, GRIDLESS Blind Source Separation method that models sources
    using a spherical Laplace distribution across frequencies.
    Separates the mixture directly without searching spatial coordinates.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, n_components=2, n_iter=20):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.n_components = n_components
        self.n_iter = n_iter

    def process(self, audio_mc, look_pos=None, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        N = min(M, self.n_components)

        # Initialize Demixing Matrices W: (F, N, M)
        W = np.zeros((F, N, M), dtype=np.complex128)
        for f in range(F):
            W[f] = np.eye(N, M, dtype=np.complex128)

        X = Zxx.transpose(1, 0, 2) # (F, M, T)
        Y = np.zeros((F, N, T), dtype=np.complex128)
        for f in range(F):
            Y[f] = W[f] @ X[f]

        # Iterative Projection (IP) for AuxIVA
        for it in range(self.n_iter):
            # Calculate source power across all frequencies for Spherical Laplace prior
            # r_n(t) = sqrt(sum_f |y_nf(t)|^2)
            power_sum = np.sum(np.abs(Y)**2, axis=0) # (N, T)
            r = np.sqrt(power_sum + 1e-12) # (N, T)

            for n in range(N):
                # The weighting (auxiliary) variable is 1 / r
                G_n = 1.0 / (2 * r[n, :] + 1e-12) # (T,)

                for f in range(F):
                    X_f = X[f] # (M, T)

                    # Weighted covariance matrix
                    U = (X_f * G_n) @ np.conj(X_f).T / T # (M, M)

                    try:
                        # IP Subspace Update (Overdetermined support)
                        # We want W[f] @ U @ W[f]^H to be diagonal.
                        A = W[f] @ U @ np.conj(W[f]).T # (N, N)
                        e_n = np.zeros(N)
                        e_n[n] = 1.0

                        x = np.linalg.solve(A, e_n) # (N,)
                        w_n_new = np.conj(W[f]).T @ x # (M,)

                        scale = np.sqrt(np.real(np.conj(w_n_new).T @ U @ w_n_new) + 1e-12)

                        W[f, n, :] = np.conj(w_n_new / scale).flatten()

                    except np.linalg.LinAlgError:
                        pass

                # Update Y immediately for next source computation
                for f in range(F):
                    Y[f, n, :] = W[f, n, :] @ X[f]

        # Target Selection (max broadband energy heuristic)
        energy_per_source = np.sum(np.abs(Y)**2, axis=(0, 2))
        target_idx = np.argmax(energy_per_source)

        # Projection Back (Scale fixing)
        Y_final = np.zeros((F, T), dtype=np.complex128)
        for f in range(F):
            try:
                W_inv = np.linalg.pinv(W[f])
                Y_final[f, :] = W_inv[0, target_idx] * Y[f, target_idx, :]
            except np.linalg.LinAlgError:
                pass

        return self.istft(Y_final)

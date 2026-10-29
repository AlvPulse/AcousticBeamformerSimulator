import numpy as np
import cvxpy as cp
from simulator.beamformers.base import BaseBeamformer

class RobustSOCP(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=0.1):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]
            R = (X_f @ np.conj(X_f).T) / T

            # Cholesky of R for the SOCP objective (min ||L^H w||)
            # Add small regularizer to R
            R_reg = R + 1e-4 * np.trace(R) * np.eye(M)
            try:
                L = np.linalg.cholesky(R_reg)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(R_reg)
                eigvals[eigvals < 0] = 0
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            a_0 = a[f, :]

            # Define CVXPY variables
            w = cp.Variable(M, complex=True)

            # Objective: min w^H R w <=> min ||L^H w||_2
            objective = cp.Minimize(cp.norm(L.conj().T @ w, 2))

            # Constraints:
            # Re{w^H a_0} >= epsilon * ||w||_2 + 1
            # Im{w^H a_0} == 0
            # Wait, the CVXPY norm is real. We can express w^H a_0
            w_H_a0 = cp.vdot(w, a_0) # cp.vdot(w, a) = w^H a

            constraints = [
                cp.real(w_H_a0) >= self.epsilon * cp.norm(w, 2) + 1,
                cp.imag(w_H_a0) == 0
            ]

            prob = cp.Problem(objective, constraints)

            try:
                prob.solve(solver=cp.SCS) # SCS supports complex SOCP
                w_opt = w.value
                if w_opt is None:
                    # Fallback to Delay and Sum if solver fails
                    w_opt = a_0 / M
            except Exception:
                w_opt = a_0 / M

            Y[f, :] = np.conj(w_opt).T @ X_f

        return self.istft(Y)

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

        # We can pre-compile the cvxpy problem by making parameters
        L_param = cp.Parameter((M, M), complex=True)
        a_0_param = cp.Parameter(M, complex=True)
        w = cp.Variable(M, complex=True)
        objective = cp.Minimize(cp.norm(L_param.conj().T @ w, 2))
        w_H_a0 = cp.vdot(w, a_0_param)
        constraints = [
            cp.real(w_H_a0) >= self.epsilon * cp.norm(w, 2) + 1,
            cp.imag(w_H_a0) == 0
        ]
        prob = cp.Problem(objective, constraints)

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

            L_param.value = L
            a_0_param.value = a[f, :]

            try:
                prob.solve(solver=cp.SCS, warm_start=True)
                w_opt = w.value
                if w_opt is None:
                    raise ValueError("Solver failed")
            except Exception as e:
                # Fallback to standard LCMV if SOCP fails
                invR = np.linalg.inv(R_reg)
                num = invR @ a[f, :]
                den = np.conj(a[f, :]).T @ num
                w_opt = num / (den + 1e-12)

            Y[f, :] = np.conj(w_opt).T @ X_f

        y_out = self.istft(Y)

        return y_out

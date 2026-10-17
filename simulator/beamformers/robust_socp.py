import numpy as np
import scipy.signal
import cvxpy as cp
from simulator.beamformers.base import BaseBeamformer

class RobustSOCP(BaseBeamformer):
    """
    Worst-Case Robust SOCP Beamformer (Vorobyov-Gershman)
    Solves min w^H R w subject to Re{w^H a_0} >= eps * ||w||_2 + 1, Im{w^H a_0} = 0
    via Second-Order Cone Programming.
    """
    def __init__(self, mismatch_eps=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mismatch_eps = mismatch_eps

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        Y = np.zeros((F, T), dtype=complex)

        # Pre-allocate cvxpy variables for speed if possible, but solving per bin is slow.
        # Fallback to DAS if solver fails.
        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :]
            R = (X_f @ np.conj(X_f.T)) / T
            # Add small loading for numerical stability in Cholesky
            R += 1e-6 * np.trace(R) * np.eye(M) / M

            a_0 = self.compute_steering_vector(array_positions, f, target_az, target_el)

            # Complex Variables in CVXPY require separate handling or CVX complex support
            w = cp.Variable(M, complex=True)

            # Objective: min w^H R w
            # Since R is PSD, we can use quad_form
            # Ensure R is exactly hermitian
            R = 0.5 * (R + np.conj(R.T))
            objective = cp.Minimize(cp.quad_form(w, R))

            # Constraints: Re{w^H a_0} >= eps * ||w||_2 + 1, Im{w^H a_0} == 0
            constraints = [
                cp.real(a_0.conj().T @ w) >= self.mismatch_eps * cp.norm(w, 2) + 1,
                cp.imag(a_0.conj().T @ w) == 0
            ]

            prob = cp.Problem(objective, constraints)

            try:
                # ECOS handles SOCP natively, SCS is also good
                prob.solve(solver=cp.SCS, verbose=False)
                if w.value is not None:
                    w_opt = w.value
                else:
                    w_opt = a_0 / M
            except cp.error.SolverError:
                w_opt = a_0 / M

            Y[f_idx, :] = np.conj(w_opt) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

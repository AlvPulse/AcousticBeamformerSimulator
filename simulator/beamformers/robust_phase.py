import numpy as np
from simulator.beamformers.base import BaseBeamformer

class PhaseOnlyLCMV(BaseBeamformer):
    """
    Gain-Invariant LCMV.
    Normalizes the input magnitude in the STFT domain for every mic at every freq bin.
    This effectively converts the signal matrix into a phase-only matrix, completely
    bypassing hardware gain miscalibrations (where one mic is 1/4th the power of another).
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        # 1. PHASE ONLY NORMALIZATION (Gain Invariant Step)
        # We replace Zxx with just its phase vectors.
        # This completely equalizes cheap, uncalibrated mics.
        mags = np.abs(Zxx) + 1e-12
        Zxx_phase = Zxx / mags

        for f in range(F):
            # Use the phase-normalized matrix for covariance calculation
            X_f_phase = Zxx_phase[:, f, :]

            # Phase-covariance matrix R: (M, M)
            R = (X_f_phase @ np.conj(X_f_phase).T) / T

            # Ledoit-Wolf Optimal Covariance Shrinkage
            trace_R = np.trace(R)
            mu = trace_R / M
            F_norm_sq = np.sum(np.abs(R - mu * np.eye(M))**2)

            var_est = 0
            for t in range(T):
                x_t = X_f_phase[:, t:t+1]
                R_t = x_t @ np.conj(x_t).T
                var_est += np.sum(np.abs(R_t - R)**2)
            var_est /= (T**2)

            if F_norm_sq > 0:
                shrinkage = max(0.0, min(1.0, float(np.real(var_est / F_norm_sq))))
            else:
                shrinkage = 1.0

            R_load = (1 - shrinkage) * R + shrinkage * mu * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_load)

            a_f = a[f, :]

            numerator = R_inv @ a_f
            denominator = np.conj(a_f).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_f / M
            else:
                w = numerator / denominator

            # Beamform using the original Zxx (so we keep the overall signal envelope)
            # but using the weights derived PURELY from Phase Covariance.
            X_f_orig = Zxx[:, f, :]
            Y[f, :] = np.conj(w).T @ X_f_orig

        y_out = self.istft(Y)
        return y_out

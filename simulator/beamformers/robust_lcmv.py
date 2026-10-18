import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class RobustLCMV(BaseBeamformer):
    def __init__(self, eps=1e-3, use_phase_stor=False, use_cmt=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.use_phase_stor = use_phase_stor
        self.use_cmt = use_cmt

    def compute_cmt_matrix(self, array_positions, freq, delta_theta=5.0):
        """ Covariance Matrix Tapering (CMT) to broaden the mainlobe. """
        M = array_positions.shape[0]
        T_mat = np.ones((M, M), dtype=complex)
        if freq == 0: return T_mat

        c = self.c
        delta_rad = np.deg2rad(delta_theta)

        # Approximate CMT by sinc function based on distance
        for i in range(M):
            for j in range(M):
                dist = np.linalg.norm(array_positions[i] - array_positions[j])
                arg = 2 * np.pi * freq * dist * np.sin(delta_rad) / c
                if arg != 0:
                    T_mat[i, j] = np.sin(arg) / arg
        return T_mat

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)

        Y = np.zeros((F, T), dtype=complex)

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :] # (M, T)

            if self.use_phase_stor:
                # Normalize magnitudes so loud amplitude spikes don't corrupt covariance
                mags = np.abs(X_f)
                mags[mags < 1e-10] = 1e-10
                X_phat = X_f / mags
                R = (X_phat @ np.conj(X_phat.T)) / T
            else:
                R = (X_f @ np.conj(X_f.T)) / T

            if self.use_cmt:
                T_mat = self.compute_cmt_matrix(array_positions, f)
                R = R * T_mat # Hadamard product

            trace_R = np.trace(R)
            load_factor = self.eps * trace_R if trace_R > 1e-10 else 1e-6
            R_load = R + load_factor * np.eye(M) / M
            a = self.compute_steering_vector(array_positions, f, target_az, target_el)

            try:
                R_inv_a = np.linalg.solve(R_load, a)
                denom = np.real(np.conj(a) @ R_inv_a)
                if denom < 1e-10:
                    w = a / M
                else:
                    w = R_inv_a / denom
            except np.linalg.LinAlgError:
                w = a / M

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

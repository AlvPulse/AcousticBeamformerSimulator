import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class LCMV_SMI(BaseBeamformer):
    def __init__(self, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        # STFT
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        # Zxx: (M, F, T)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)

        Y = np.zeros((F, T), dtype=complex)

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :] # (M, T)

            # Covariance matrix R
            R = (X_f @ np.conj(X_f.T)) / T

            # Diagonal Loading
            R_load = R + self.eps * np.trace(R) * np.eye(M) / M

            # Steering vector
            a = self.compute_steering_vector(array_positions, f, target_az, target_el)

            # w = R^-1 a / (a^H R^-1 a)
            try:
                R_inv_a = np.linalg.solve(R_load, a)
                denominator = np.conj(a) @ R_inv_a
                w = R_inv_a / np.real(denominator)
            except np.linalg.LinAlgError:
                w = a / M # Fallback to DAS

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

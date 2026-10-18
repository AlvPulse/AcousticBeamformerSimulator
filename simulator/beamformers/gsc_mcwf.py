import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class GSC_MCWF(BaseBeamformer):
    """
    Generalized Sidelobe Canceller (GSC) with Multi-Channel Wiener Filter (MCWF) Post-Filtering.
    Uses Zelinsky post-filter estimator to clean residual noise based on the variance
    of the noise reference from the blocking matrix, ensuring optimal MSE without musical noise.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        Y = np.zeros((F, T), dtype=complex)

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :] # (M, T)

            # 1. Fixed Beamformer (DAS) path
            a = self.compute_steering_vector(array_positions, f, target_az, target_el)
            w_q = a / M
            y_fbf = np.conj(w_q) @ X_f # Target + Noise (1, T)

            # 2. Blocking Matrix (Griffiths-Jim)
            # Find an orthogonal matrix B such that B^H a = 0
            B = np.zeros((M-1, M), dtype=complex)
            for m in range(M-1):
                B[m, 0] = a[m+1]
                B[m, m+1] = -a[0]

            # Normalize rows
            for m in range(M-1):
                B[m] = B[m] / (np.linalg.norm(B[m]) + 1e-10)

            # Applying B directly (not conjugate) since B a = 0 is designed, not B^H a = 0
            u = B @ X_f # Noise references (M-1, T)

            # 3. Adaptive Interference Canceller (GSC part)
            R_uu = (u @ np.conj(u.T)) / T
            # Add robust diagonal loading to prevent singular matrix in silent frames
            trace_R_uu = np.trace(R_uu)
            load_factor = 1e-4 * trace_R_uu if trace_R_uu > 1e-10 else 1e-6
            R_uu += load_factor * np.eye(M-1) / (M-1)
            r_uy = (u @ np.conj(y_fbf.T)) / T

            try:
                w_a = np.linalg.solve(R_uu, r_uy)
                y_gsc = y_fbf - np.conj(w_a) @ u
            except np.linalg.LinAlgError:
                # Safe fallback to fixed beamformer if adaptive step completely fails
                y_gsc = y_fbf

            # 4. Multi-Channel Wiener Post-Filter (Zelinsky Estimator)
            Phi_yy = np.mean(np.abs(y_gsc)**2)
            Phi_nn = np.mean(np.abs(u)**2) # average across sensors and time

            # Safe bounding of the Wiener gain to prevent zeroing out or blowing up
            if Phi_yy > 1e-10:
                # Limit the maximum suppression to -20dB (0.01) to avoid musical noise artifacts
                H_wiener = max(0.01, 1.0 - (Phi_nn / Phi_yy))
            else:
                H_wiener = 0.01

            Y[f_idx, :] = y_gsc * H_wiener

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

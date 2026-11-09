import numpy as np
import scipy.signal as signal
from simulator.beamformers.base import BaseBeamformer

class INCM_MVDR(BaseBeamformer):
    """
    Interference-plus-Noise Covariance Matrix based MVDR (INCM-MVDR).
    A highly robust grid-based beamformer. Instead of using the spatial covariance
    of the entire signal (which causes target self-cancellation under steering mismatch),
    it estimates the covariance of ONLY the interference and noise.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        # To build the INCM, we need to separate Target frames from Noise frames.
        # We use a simple energy-based Voice/Target Activity Detector.
        # Frames with energy below the 40th percentile are considered pure Noise/Interference.
        frame_energies = np.sum(np.abs(Zxx)**2, axis=(0, 1))
        noise_threshold = np.percentile(frame_energies, 40)
        noise_mask = frame_energies < noise_threshold

        # Failsafe if the environment is too continuous
        if np.sum(noise_mask) < 2:
            noise_mask = np.ones(T, dtype=bool)

        for f in range(F):
            X_f = Zxx[:, f, :]

            # Extract only the noise/interference frames for the covariance matrix
            X_noise = X_f[:, noise_mask]

            # Interference-plus-Noise Covariance Matrix (INCM)
            R_in = (X_noise @ np.conj(X_noise).T) / max(1, X_noise.shape[1])

            # Apply diagonal loading to ensure matrix is invertible and robust
            R_load = R_in + self.epsilon * np.trace(R_in) * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_load)

            a_f = a[f, :]

            # MVDR weights based purely on the INCM
            numerator = R_inv @ a_f
            denominator = np.conj(a_f).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_f / M
            else:
                w = numerator / denominator

            # Apply the weights to the FULL signal to extract the target
            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

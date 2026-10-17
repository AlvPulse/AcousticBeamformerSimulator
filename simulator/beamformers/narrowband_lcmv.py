import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class NarrowbandLCMV(BaseBeamformer):
    """
    Time-domain / Single-bin constrained Narrowband LCMV.
    Calculates one spatial covariance matrix across the *entire* broadband signal,
    and calculates one weight vector based purely on the time-delay steering vector
    (or center frequency).
    This highlights why wideband STFT-based LCMV is required for broadband audio.
    """
    def __init__(self, eps=1e-3, center_freq=1000.0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.center_freq = center_freq

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        # multi_channel_audio: (M, T)
        M, T = multi_channel_audio.shape

        # 1. Compute single broadband spatial covariance matrix
        # (This is wrong for broadband signals as phase shifts are frequency dependent)
        R = (multi_channel_audio @ multi_channel_audio.T) / T

        # Diagonal loading
        R_load = R + self.eps * np.trace(R) * np.eye(M) / M

        # 2. Compute steering vector at a single center frequency
        a = self.compute_steering_vector(array_positions, self.center_freq, target_az, target_el)

        # 3. Compute narrowband weights
        try:
            R_inv_a = np.linalg.solve(R_load, a)
            denominator = np.conj(a) @ R_inv_a
            w = R_inv_a / np.real(denominator)
        except np.linalg.LinAlgError:
            w = a / M

        # 4. Apply weights directly to time-domain signal (as if it was narrowband)
        # Note: This just applies complex scalar weights to the real signal.
        # We take the real part of the output.
        y_out = np.real(np.conj(w) @ multi_channel_audio)

        return y_out

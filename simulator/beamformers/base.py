import numpy as np
import scipy.signal

class BaseBeamformer:
    def __init__(self, n_fft=512, hop_length=256, fs=16000, c=343.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fs = fs
        self.c = c
        self.window = scipy.signal.windows.hann(n_fft)

    def compute_steering_vector(self, array_positions, freq, az_deg, el_deg):
        """
        Computes the far-field steering vector for a given frequency and direction.
        array_positions: (M, 3)
        freq: scalar (Hz)
        az_deg, el_deg: scalars (degrees)
        Returns: (M,) complex array
        """
        if freq == 0:
            return np.ones(array_positions.shape[0], dtype=complex)

        az_rad = np.deg2rad(az_deg)
        el_rad = np.deg2rad(el_deg)

        k_vec = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        tau = -np.dot(array_positions, k_vec) / self.c
        a = np.exp(-1j * 2 * np.pi * freq * tau)
        return a

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        """
        Abstract method. Must return a single channel audio signal.
        """
        raise NotImplementedError

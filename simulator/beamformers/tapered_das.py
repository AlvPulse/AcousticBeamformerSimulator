import numpy as np
import scipy.signal.windows as windows
from simulator.beamformers.base import BaseBeamformer

class TaperedDAS(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, sidelobe_att_db=50):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.sidelobe_att_db = sidelobe_att_db

        # Chebyshev spatial window for 50 elements
        self.spatial_window = windows.chebwin(self.n_mics, at=self.sidelobe_att_db)
        self.spatial_window /= np.sum(self.spatial_window)

    def process(self, audio_mc, look_pos, **kwargs):
        # 1. Compute STFT
        Zxx = self.stft(audio_mc) # (M, F, T)

        # 2. Compute Steering Vector
        a = self.compute_steering_vector(look_pos) # (F, M)

        # 3. Apply weights
        # w = a * spatial_window
        w = a * self.spatial_window[np.newaxis, :] # (F, M)

        # 4. Beamform
        # Zxx is (M, F, T) -> transpose to (F, M, T)
        Zxx_fmt = np.transpose(Zxx, (1, 0, 2))

        # w is (F, M), we want w^H * X
        # for each freq f: w[f]^H @ Zxx_fmt[f]
        w_conj = np.conj(w) # (F, M)

        # Multiply and sum over mics
        Y = np.sum(w_conj[:, :, np.newaxis] * Zxx_fmt, axis=1) # (F, T)

        # 5. Inverse STFT
        y_out = self.istft(Y)
        return y_out

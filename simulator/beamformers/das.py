import numpy as np
from simulator.beamformers.base import BaseBeamformer

class StandardDAS(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256):
        super().__init__(pos_mics, fs, n_fft, hop_length)

    def process(self, audio_mc, look_pos, **kwargs):
        # 1. Compute STFT
        Zxx = self.stft(audio_mc) # (M, F, T)

        # 2. Compute Steering Vector
        a = self.compute_steering_vector(look_pos) # (F, M)

        # 3. Beamform
        Zxx_fmt = np.transpose(Zxx, (1, 0, 2)) # (F, M, T)

        # w is just a, we want w^H * X
        w_conj = np.conj(a) # (F, M)

        # Multiply and sum over mics
        Y = np.sum(w_conj[:, :, np.newaxis] * Zxx_fmt, axis=1) # (F, T)

        # 4. Inverse STFT
        return self.istft(Y)

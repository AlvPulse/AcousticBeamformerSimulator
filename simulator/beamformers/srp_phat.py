import numpy as np
import scipy.signal as signal
from simulator.beamformers.base import BaseBeamformer

class SRP_PHAT(BaseBeamformer):
    """
    Steered Response Power with Phase Transform (SRP-PHAT).
    A robust classical method that computes generalized cross-correlation (GCC)
    with phase-whitening to localize sources in reverberant environments.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256):
        super().__init__(pos_mics, fs, n_fft, hop_length)

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]

            # Phase Transform (PHAT) - whiten the spectrum by dividing by magnitude
            # This makes the method robust to amplitude variations and reverberation
            mag = np.abs(X_f)
            X_f_phat = X_f / (mag + 1e-12)

            # Steer the phase-whitened signals (Delay and Sum on Phase)
            a_f = a[f, :]
            w = a_f / M

            Y[f, :] = np.conj(w).T @ X_f_phat

        return self.istft(Y)

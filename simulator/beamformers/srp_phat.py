import numpy as np
from simulator.beamformers.base import BaseBeamformer

class SRP_PHAT(BaseBeamformer):
    """
    Steered Response Power with Phase Transform (SRP-PHAT).
    A highly robust classical method that normalizes the magnitude of the signal
    before beamforming, relying purely on phase. It is mathematically equivalent to
    PhaseOnlyLCMV without the matrix inversion, making it a pure Delay-and-Sum on Phase.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256):
        super().__init__(pos_mics, fs, n_fft, hop_length)

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        # Phase Transform (PHAT) - Normalize magnitudes
        mags = np.abs(Zxx) + 1e-12
        Zxx_phat = Zxx / mags

        for f in range(F):
            # Pure Delay and Sum on the phase-whitened signals
            # Y = a^H * X_phat
            a_f = a[f, :]

            # Because we destroyed the amplitude envelope, we must restore it
            # by projecting the phase-aligned signal back onto the original signal amplitude.
            # A common approximation for SRP-PHAT audio extraction is to use the raw DAS
            # but weighted by the SRP coherence.

            y_phat = np.conj(a_f).T @ Zxx_phat[:, f, :]
            coherence_weight = np.abs(y_phat) / M

            # Standard DAS
            y_das = np.conj(a_f).T @ Zxx[:, f, :]

            Y[f, :] = y_das * coherence_weight

        return self.istft(Y)

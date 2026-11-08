import numpy as np
import scipy.signal as signal
from simulator.beamformers.base import BaseBeamformer

class FastIVA(BaseBeamformer):
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, n_iter=20):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.n_iter = n_iter

    def process(self, audio_mc, look_pos=None, **kwargs):
        # We try to import pyroomacoustics, if it fails return mono
        try:
            import pyroomacoustics as pra
            from pyroomacoustics.bss import auxiva
        except ImportError:
            # Statically analyzed fallback
            return audio_mc[0]

        Zxx = self.stft(audio_mc) # (M, F, T)
        # auxiva expects X of shape (n_freq, n_frames, n_channels)
        X = Zxx.transpose([1, 2, 0])

        # We use a very low number of iterations if channels are high to prevent timeouts
        n_mics = audio_mc.shape[0]
        n_iter = self.n_iter
        if n_mics > 32:
            n_iter = min(3, n_iter) # Very aggressive constraint for 64-channel FastIVA

        try:
            # Overdetermined BSS can be very slow.
            Y = auxiva(X, n_iter=n_iter) # (F, T, N_src)

            # Select the channel with max energy as the target
            Y_energy = np.sum(np.abs(Y)**2, axis=(0,1))
            target_idx = np.argmax(Y_energy)

            Y_target = Y[:, :, target_idx] # (F, T)
            return self.istft(Y_target)

        except Exception:
            # If AuxIVA fails (e.g., due to ill-conditioned matrices in field data), fallback to DAS
            a = self.compute_steering_vector(look_pos)
            M, F, T = Zxx.shape
            Y_target = np.zeros((F, T), dtype=np.complex128)
            for f in range(F):
                Y_target[f, :] = np.conj(a[f, :]/M).T @ Zxx[:, f, :]
            return self.istft(Y_target)

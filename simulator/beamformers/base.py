import numpy as np
import scipy.signal as signal

class BaseBeamformer:
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256):
        self.pos_mics = pos_mics
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.c = 343.0

        # Precompute frequencies
        self.freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        self.freqs[0] = 1e-6  # Avoid division by zero
        self.n_freqs = len(self.freqs)
        self.n_mics = pos_mics.shape[0]

    def compute_steering_vector(self, look_pos, is_spherical=False):
        """
        Computes steering vectors for all frequencies.
        Returns: (n_freqs, n_mics)
        """
        from simulator.acoustics.channel import compute_delay_and_attenuation
        delays, _ = compute_delay_and_attenuation(self.pos_mics, look_pos, self.c, is_spherical)

        # delays is (n_mics,)
        # freqs is (n_freqs,)
        # a = exp(-j * 2 * pi * f * delay)
        # However, to steer, we apply +j * 2 * pi * f * delay (conjugate)

        omega = 2 * np.pi * self.freqs
        phase = np.outer(omega, delays) # (n_freqs, n_mics)
        a = np.exp(1j * phase)

        # Normalize
        a = a / np.sqrt(self.n_mics)
        return a

    def process(self, audio_mc, look_pos, **kwargs):
        """
        Takes multi-channel audio (n_mics, n_samples) and returns beamformed mono (n_samples,).
        """
        raise NotImplementedError("Process must be implemented by subclasses.")

    def stft(self, audio_mc):
        """
        Returns STFT matrix (n_mics, n_freqs, n_frames)
        """
        f, t, Zxx = signal.stft(audio_mc, fs=self.fs, nperseg=self.n_fft,
                                noverlap=self.n_fft - self.hop_length,
                                window='hann', return_onesided=True)
        return Zxx

    def istft(self, Zxx):
        """
        Returns time domain signal (n_samples,)
        """
        t, x = signal.istft(Zxx, fs=self.fs, nperseg=self.n_fft,
                            noverlap=self.n_fft - self.hop_length,
                            window='hann', input_onesided=True)
        return x

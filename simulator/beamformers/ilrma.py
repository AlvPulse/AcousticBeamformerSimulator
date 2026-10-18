import numpy as np
import scipy.signal
import pyroomacoustics as pra
from simulator.beamformers.base import BaseBeamformer

class ILRMA(BaseBeamformer):
    """
    Independent Low-Rank Matrix Analysis (ILRMA).
    Blind Source Separation method combining FastICA with NMF to separate sources
    based on the statistical structure (low rank) of their spectrograms, requiring zero labeled data.
    """
    def __init__(self, n_sources=2, n_iter=30, n_components=5, **kwargs):
        super().__init__(**kwargs)
        self.n_sources = n_sources
        self.n_iter = n_iter
        self.n_components = n_components

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)

        # Current Zxx is (M, F, T) -> Need (F, T, M)
        X = np.transpose(Zxx, (1, 2, 0))

        # Fallback to DAS if there are fewer mics than requested sources
        if X.shape[2] < self.n_sources:
             from .tapered_das import TaperedDAS
             das = TaperedDAS(fs=self.fs)
             return das.process(multi_channel_audio, array_positions, target_az, target_el)

        try:
            Y = pra.bss.ilrma(X, n_src=self.n_sources, n_iter=self.n_iter,
                              proj_back=True, n_components=self.n_components)

            M = array_positions.shape[0]
            freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)

            best_source_idx = 0
            max_spatial_power = -np.inf

            for s in range(self.n_sources):
                source_stft = Y[:, :, s]
                energy = np.sum(np.abs(source_stft)**2)

                if energy > max_spatial_power:
                    max_spatial_power = energy
                    best_source_idx = s

            best_stft = Y[:, :, best_source_idx]

            _, y_out = scipy.signal.istft(best_stft, fs=self.fs, window=self.window,
                                          nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
            return y_out

        except Exception as e:
            # Fallback to DAS if ILRMA fails to converge
            from .tapered_das import TaperedDAS
            das = TaperedDAS(fs=self.fs)
            return das.process(multi_channel_audio, array_positions, target_az, target_el)

import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class TaperedDAS(BaseBeamformer):
    def __init__(self, sidelobe_attenuation_db=40, **kwargs):
        super().__init__(**kwargs)
        self.sidelobe_db = sidelobe_attenuation_db

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        M = array_positions.shape[0]

        # Chebyshev spatial taper
        # For a random/spiral array, standard 1D Chebyshev doesn't map perfectly,
        # but we can apply a proxy mapping by distance from center, or just use
        # a standard window on the array elements sorted by radius.
        radii = np.linalg.norm(array_positions, axis=1)
        sort_idx = np.argsort(radii)

        # Standard 1D chebwin applied radially
        taper_1d = scipy.signal.windows.chebwin(M, at=self.sidelobe_db)
        spatial_weights = np.zeros(M)
        spatial_weights[sort_idx] = taper_1d
        spatial_weights /= np.sum(spatial_weights) # Normalize sum to 1

        # STFT
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        # Zxx: (M, F, T)

        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)

        Y = np.zeros((Zxx.shape[1], Zxx.shape[2]), dtype=complex)

        for f_idx, f in enumerate(freqs):
            a = self.compute_steering_vector(array_positions, f, target_az, target_el)
            # DAS weight: w = a / M  (we also apply spatial tapering)
            w = spatial_weights * a

            # w^H * x
            Y[f_idx, :] = np.conj(w) @ Zxx[:, f_idx, :]

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

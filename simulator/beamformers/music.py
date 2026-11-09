import numpy as np
from simulator.beamformers.base import BaseBeamformer

class MUSIC(BaseBeamformer):
    """
    MUltiple SIgnal Classification (MUSIC).
    A high-resolution classical subspace method. Separates the covariance matrix
    into Signal and Noise subspaces to compute a sharp pseudo-spectrum.
    Because MUSIC only outputs a spatial spectrum (not an audio signal),
    we synthesize an output by weighting the input signal by the MUSIC spectrum peak.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, n_sources=1):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.n_sources = n_sources

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        for f in range(F):
            X_f = Zxx[:, f, :]
            R = (X_f @ np.conj(X_f).T) / T

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(R)

            # Sort eigenvalues descending
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]

            # Noise Subspace (E_N)
            E_N = eigvecs[:, self.n_sources:]

            # MUSIC Pseudo-spectrum at the look direction: P_music = 1 / (a^H * E_N * E_N^H * a)
            a_f = a[f, :]
            projection = np.conj(a_f).T @ E_N @ np.conj(E_N).T @ a_f

            # P_music acts as a spatial probability or gain for this direction
            p_music = 1.0 / (np.abs(projection) + 1e-12)

            # To output audio (since MUSIC isn't a standard beamformer), we apply the pseudo-spectrum
            # as a gain to the delay-and-summed signal.
            y_das = np.conj(a_f).T @ X_f

            # We clip the gain to prevent massive explosions when projection is near 0
            gain = min(p_music, 100.0)
            Y[f, :] = y_das * (gain / 100.0)

        return self.istft(Y)

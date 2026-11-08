import numpy as np
import scipy.signal as signal
from simulator.beamformers.base import BaseBeamformer

class MUSIC(BaseBeamformer):
    """
    Multiple Signal Classification (MUSIC).
    A high-resolution classical subspace method for Direction of Arrival (DOA) estimation.
    As a DOA estimator, it computes a pseudo-spectrum rather than a filtered audio signal.
    For pipeline compatibility, the output audio is synthesized such that its energy
    matches the MUSIC pseudo-spectrum at the look direction.
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

            # Spatial Covariance Matrix
            R = (X_f @ np.conj(X_f).T) / T

            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(R)

            # Sort eigenvalues descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # Separate noise subspace
            # If n_sources is known or estimated, noise subspace starts from index n_sources
            if M > self.n_sources:
                En = eigvecs[:, self.n_sources:]
            else:
                En = eigvecs # Failsafe

            # MUSIC pseudo-spectrum at this frequency and look direction
            a_f = a[f, :]
            # P_music = 1 / (a^H * En * En^H * a)
            denom = np.abs(np.conj(a_f).T @ En @ np.conj(En).T @ a_f)
            p_music = 1.0 / (denom + 1e-12)

            # Synthesize output: we pass the strongest microphone signal scaled by the normalized MUSIC spectrum
            # This ensures the output power maps directly to the spatial likelihood for the heatmap.
            # We apply a bounded scaling to prevent explosion
            scale = np.sqrt(np.clip(p_music, 0, 100))

            # We use DAS as the carrier signal, scaled by MUSIC pseudo-spectrum
            w_das = a_f / M
            Y[f, :] = (np.conj(w_das).T @ X_f) * scale

        return self.istft(Y)

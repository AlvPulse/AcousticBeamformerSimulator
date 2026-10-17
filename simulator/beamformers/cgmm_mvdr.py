import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer

class cGMM_MVDR(BaseBeamformer):
    """
    Mask-Based (cGMM) MVDR Benchmark
    Estimates Target Covariance R_ss and Noise Covariance R_nn
    using ideal masks (approximated here by SNR energy thresholds for benchmarking)
    and solves the generalized MVDR.
    """
    def __init__(self, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        Y = np.zeros((F, T), dtype=complex)

        # Calculate global energy per frame for heuristic mask
        # In a real cGMM, this is derived from EM clustering on spatial features.
        # For benchmarking limits, an energy-based Voice Activity Detection (VAD) proxy is used.
        frame_energy = np.sum(np.abs(Zxx)**2, axis=(0, 1))
        thresh = np.median(frame_energy)

        target_mask = (frame_energy > thresh).astype(float)
        noise_mask = 1.0 - target_mask

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :] # (M, T)

            # Weighted Covariance Matrices
            R_ss = (X_f * target_mask) @ np.conj((X_f * target_mask).T) / (np.sum(target_mask) + 1e-8)
            R_nn = (X_f * noise_mask) @ np.conj((X_f * noise_mask).T) / (np.sum(noise_mask) + 1e-8)

            # Diagonal loading on noise covariance
            R_nn_load = R_nn + self.eps * np.trace(R_nn) * np.eye(M) / M

            # Estimate Principal Eigenvector of R_ss as the steered vector
            # Alternatively, standard generalized principal eigenvector of (R_nn^-1 R_ss)
            eigvals, eigvecs = np.linalg.eigh(R_ss)
            a_hat = eigvecs[:, -1] # Principal eigenvector

            # Align phase with nominal steering vector to prevent arbitrary phase flips
            a_0 = self.compute_steering_vector(array_positions, f, target_az, target_el)
            phase_diff = np.exp(1j * np.angle(np.conj(a_hat) @ a_0))
            a_hat = a_hat * phase_diff

            try:
                R_inv_a = np.linalg.solve(R_nn_load, a_hat)
                denominator = np.conj(a_hat) @ R_inv_a
                w = R_inv_a / np.real(denominator)
            except np.linalg.LinAlgError:
                w = a_0 / M

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

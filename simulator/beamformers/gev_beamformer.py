import numpy as np
import scipy.signal
import scipy.linalg
from simulator.beamformers.base import BaseBeamformer

class GEVBeamformer(BaseBeamformer):
    """
    Generalized Eigenvalue (GEV) Beamformer.
    Maximizes SNR by finding the principal generalized eigenvector of (R_ss, R_nn).
    Does NOT rely on a physical steering vector, making it highly robust to
    array calibration errors and wavefront warping.
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

        # Use energy mask as a proxy for Voice Activity Detection (VAD)
        # to separate Target+Noise frames from pure Noise frames.
        frame_energy = np.sum(np.abs(Zxx)**2, axis=(0, 1))
        thresh = np.median(frame_energy)

        target_mask = (frame_energy > thresh).astype(float)
        noise_mask = 1.0 - target_mask

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :]

            # Estimate covariances
            R_ss = (X_f * target_mask) @ np.conj((X_f * target_mask).T) / (np.sum(target_mask) + 1e-8)
            R_nn = (X_f * noise_mask) @ np.conj((X_f * noise_mask).T) / (np.sum(noise_mask) + 1e-8)

            # Diagonal loading on noise covariance to ensure positive definite
            R_nn_load = R_nn + self.eps * np.trace(R_nn) * np.eye(M) / M

            # Solve Generalized Eigenvalue Problem: R_ss * w = lambda * R_nn * w
            try:
                # scipy.linalg.eigh solves generalized eigenvalue problem A x = lambda B x
                # It returns eigenvalues and normalized eigenvectors
                eigvals, eigvecs = scipy.linalg.eigh(R_ss, R_nn_load)

                # The optimal GEV weights are the eigenvector corresponding to the largest eigenvalue
                w = eigvecs[:, -1]

                # GEV introduces arbitrary phase scaling.
                # Blind phase alignment: align to the reference mic (index 0)
                phase_correction = np.exp(-1j * np.angle(w[0]))
                w = w * phase_correction

            except scipy.linalg.LinAlgError:
                # Fallback to DAS if matrices are extremely ill-conditioned
                w = self.compute_steering_vector(array_positions, f, target_az, target_el) / M

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

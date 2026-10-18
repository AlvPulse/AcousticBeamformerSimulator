import numpy as np
import scipy.signal
from simulator.beamformers.base import BaseBeamformer
from simulator.beamformers.lcmv_smi import LCMV_SMI
from simulator.beamformers.tapered_das import TaperedDAS

class SpatialVarianceMask(BaseBeamformer):
    """
    Statistical Time-Frequency Masking using Inter-Channel Phase Difference (IPD) Variance.
    Dynamically switches between LCMV (for coherent signals like cars) and
    DAS (for incoherent noise like wind) per frequency bin based on phase stability.
    """
    def __init__(self, variance_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.variance_thresh = variance_threshold
        self.lcmv = LCMV_SMI(**kwargs)
        self.das = TaperedDAS(**kwargs)

    def process(self, multi_channel_audio, array_positions, target_az, target_el):
        _, _, Zxx = scipy.signal.stft(multi_channel_audio, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length,
                                      axis=-1)
        M, F, T = Zxx.shape
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.fs)
        Y = np.zeros((F, T), dtype=complex)

        for f_idx, f in enumerate(freqs):
            X_f = Zxx[:, f_idx, :] # (M, T)

            # Calculate Inter-Channel Phase Difference (IPD) Variance
            phases = np.angle(X_f)
            ref_phase = phases[0, :]
            ipd = np.mod(phases - ref_phase + np.pi, 2*np.pi) - np.pi
            ipd_var = np.mean(np.var(ipd, axis=0))

            a = self.compute_steering_vector(array_positions, f, target_az, target_el)

            # Safe NaN check in case of absolute silence
            if np.isnan(ipd_var) or ipd_var > self.variance_thresh:
                # Incoherent / Wind / Diffuse noise or Silent frame.
                # Adaptive methods will self-cancel. Fall back to passive Tapered DAS.
                taper_1d = scipy.signal.windows.chebwin(M, at=40)
                radii = np.linalg.norm(array_positions, axis=1)
                sort_idx = np.argsort(radii)
                spatial_weights = np.zeros(M)
                spatial_weights[sort_idx] = taper_1d
                spatial_weights /= np.sum(spatial_weights)
                w = spatial_weights * a
            else:
                # Coherent wave (target or near-field car). Safe to use adaptive LCMV.
                R = (X_f @ np.conj(X_f.T)) / T
                trace_R = np.trace(R)
                load_factor = 1e-3 * trace_R if trace_R > 1e-10 else 1e-6
                R_load = R + load_factor * np.eye(M) / M

                try:
                    R_inv_a = np.linalg.solve(R_load, a)
                    denom = np.real(np.conj(a) @ R_inv_a)
                    # Prevent divide by zero in extreme cases
                    if denom < 1e-10:
                        w = a / M
                    else:
                        w = R_inv_a / denom
                except np.linalg.LinAlgError:
                    w = a / M

            Y[f_idx, :] = np.conj(w) @ X_f

        _, y_out = scipy.signal.istft(Y, fs=self.fs, window=self.window,
                                      nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length)
        return y_out

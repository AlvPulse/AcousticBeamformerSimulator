import numpy as np
from simulator.beamformers.base import BaseBeamformer

class INCM_MVDR(BaseBeamformer):
    """
    Interference-plus-Noise Covariance Matrix (INCM) MVDR.
    A highly robust grid-based beamformer that mitigates target self-cancellation
    by explicitly removing the target's covariance from the data covariance matrix
    before computing the MVDR weights.
    """
    def __init__(self, pos_mics, fs, n_fft=512, hop_length=256, epsilon=1e-3):
        super().__init__(pos_mics, fs, n_fft, hop_length)
        self.epsilon = epsilon

    def process(self, audio_mc, look_pos, **kwargs):
        Zxx = self.stft(audio_mc) # (M, F, T)
        M, F, T = Zxx.shape

        # Theoretical steering vector
        a = self.compute_steering_vector(look_pos) # (F, M)
        Y = np.zeros((F, T), dtype=np.complex128)

        # Estimate Target presence (Simple VAD based on energy) to help isolate the target
        frame_energies = np.sum(np.abs(Zxx)**2, axis=(0, 1))
        p_target = np.percentile(frame_energies, 80)
        target_mask = frame_energies > p_target

        if np.sum(target_mask) < 2:
            target_mask = np.ones(T, dtype=bool)

        for f in range(F):
            X_f = Zxx[:, f, :] # (M, T)

            # Full Data Covariance Matrix
            R_xx = (X_f @ np.conj(X_f).T) / T

            # Target Covariance Estimation (Spatial Subspace Projection)
            # 1. We assume the target is present in the high-energy frames
            X_target = X_f[:, target_mask]
            R_target_raw = (X_target @ np.conj(X_target).T) / max(1, X_target.shape[1])

            # 2. Extract the target signal component using the nominal steering vector
            a_f = a[f, :]
            # Eigenspace projection to find the actual target vector (SVE step to be robust)
            try:
                eigvals, eigvecs = np.linalg.eigh(R_target_raw)
                a_actual = eigvecs[:, -1]
                # Align phase with nominal vector
                phase_align = np.exp(-1j * np.angle(np.conj(a_f).T @ a_actual))
                a_actual = a_actual * phase_align
                a_actual = a_actual / (np.linalg.norm(a_actual) + 1e-12) * np.sqrt(M)
            except np.linalg.LinAlgError:
                a_actual = a_f

            # 3. Estimate target variance
            var_s = np.real(np.conj(a_actual).T @ R_target_raw @ a_actual) / (M**2)

            # 4. Construct Target Covariance Matrix
            R_target = var_s * np.outer(a_actual, np.conj(a_actual))

            # 5. Form the INCM (Interference + Noise Covariance Matrix)
            R_incm = R_xx - R_target

            # Ensure R_incm is positive semi-definite (clip negative eigenvalues)
            try:
                eigvals, eigvecs = np.linalg.eigh(R_incm)
                eigvals[eigvals < 0] = 1e-12
                R_incm = eigvecs @ np.diag(eigvals) @ np.conj(eigvecs).T
            except np.linalg.LinAlgError:
                pass

            # Diagonal loading for robustness
            R_load = R_incm + self.epsilon * np.trace(R_incm) * np.eye(M)

            try:
                R_inv = np.linalg.inv(R_load)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_load)

            # MVDR Weights using the INCM and actual estimated target vector
            numerator = R_inv @ a_actual
            denominator = np.conj(a_actual).T @ numerator

            if np.abs(denominator) < 1e-12:
                w = a_actual / M
            else:
                w = numerator / denominator

            # Beamform
            Y[f, :] = np.conj(w).T @ X_f

        return self.istft(Y)

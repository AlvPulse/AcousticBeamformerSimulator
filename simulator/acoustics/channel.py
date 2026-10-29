import numpy as np
import scipy.special as sp

def compute_delay_and_attenuation(pos_mics, source_pos, c=343.0, is_spherical=True):
    """
    Computes time delays and distance attenuations for mics relative to origin.
    pos_mics: (M, 3) array of mic coordinates.
    source_pos: (3,) source coordinate.
    is_spherical: True for 1/r and exact distance, False for plane wave approximation.
    Returns:
        delays: (M,) in seconds.
        attenuations: (M,) linear scale factor.
    """
    M = pos_mics.shape[0]
    if is_spherical:
        distances = np.linalg.norm(pos_mics - source_pos, axis=1)
        delays = distances / c
        attenuations = 1.0 / (4 * np.pi * distances)
        # Normalize attenuation relative to array center to prevent massive overall volume drop
        center_dist = np.linalg.norm(source_pos)
        center_att = 1.0 / (4 * np.pi * center_dist)
        attenuations = attenuations / center_att
    else:
        # Plane wave
        # Unit vector pointing FROM origin TO source
        u = source_pos / np.linalg.norm(source_pos)
        # Delays relative to origin
        delays = -np.dot(pos_mics, u) / c
        attenuations = np.ones(M)

    return delays, attenuations

def generate_wind_diffuse_noise(pos_mics, n_samples, fs, c=343.0):
    """
    Generates spatially diffuse noise via cylindrical isotropic coherence matrix (J0).
    Adds -6 dB/octave low-frequency rumble profile.
    """
    M = pos_mics.shape[0]
    n_fft = 1024
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    freqs[0] = 1e-6 # prevent div by zero

    # Precompute pairwise distances
    dist_mat = np.linalg.norm(pos_mics[:, None, :] - pos_mics[None, :, :], axis=2)

    # Generate white noise in frequency domain
    Z = np.random.randn(M, len(freqs)) + 1j * np.random.randn(M, len(freqs))

    X_diffuse = np.zeros_like(Z)

    for k, f in enumerate(freqs):
        # Cylindrical isotropic coherence
        arg = 2 * np.pi * f * dist_mat / c
        Gamma = sp.jv(0, arg) # J0 Bessel function

        # Cholesky factorization needs small regularization for numerical stability
        Gamma += np.eye(M) * 1e-6
        try:
            L = np.linalg.cholesky(Gamma)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(Gamma)
            eigvals[eigvals < 0] = 0
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Shape with -6 dB/octave rumble profile (1/f)
        rumble_gain = 1.0 / f

        X_diffuse[:, k] = L @ Z[:, k] * rumble_gain

    # IFFT
    noise_time = np.fft.irfft(X_diffuse, n=n_fft, axis=1)

    # Repeat or tile to desired length
    repeats = int(np.ceil(n_samples / n_fft))
    noise_time = np.tile(noise_time, (1, repeats))[:, :n_samples]

    return noise_time

def apply_turbulence_decoherence(stft_matrix, pos_mics, source_pos, freqs, sigma_phi_base=0.1):
    """
    Perturbs phase of target to mimic thermal/wind eddies.
    stft_matrix: (M, F, T)
    sigma_phi(f, d) = alpha * f * sqrt(d)
    """
    M, F, T = stft_matrix.shape
    d = np.linalg.norm(source_pos)

    # alpha calculated such that at max freq and 120m, max jitter matches sigma_phi_base
    alpha = sigma_phi_base / (freqs[-1] * np.sqrt(d) + 1e-6)

    sigma_phi = alpha * freqs * np.sqrt(d) # Shape: (F,)

    # Generate phase jitter for each mic, freq, and time
    # (M, F, T)
    jitter = np.random.randn(M, F, T) * sigma_phi[None, :, None]

    stft_matrix_perturbed = stft_matrix * np.exp(1j * jitter)
    return stft_matrix_perturbed

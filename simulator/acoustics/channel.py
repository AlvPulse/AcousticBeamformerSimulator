import numpy as np
import scipy.special as sp

c = 343.0 # Speed of sound (m/s)

def apply_greens_function(source_signal, fs, source_pos, mic_positions):
    """
    Applies exact spherical wave Green's function for near-field attenuation and delay.
    source_signal: (n_samples,)
    source_pos: (3,)
    mic_positions: (n_sensors, 3)
    Returns: (n_sensors, n_samples)
    """
    n_sensors = mic_positions.shape[0]
    n_samples = len(source_signal)

    # Calculate distances
    distances = np.linalg.norm(mic_positions - source_pos, axis=1) # (n_sensors,)

    # Attenuation 1 / (4 * pi * r)
    attenuation = 1.0 / (4 * np.pi * distances)

    # Delays in samples
    delays_sec = distances / c
    delays_samples = delays_sec * fs

    # To prevent circular convolution wrap-around when shifting,
    # zero-pad to length N + max_delay
    max_delay_samples = int(np.max(delays_samples)) + 1
    padded_len = n_samples + max_delay_samples

    output = np.zeros((n_sensors, padded_len))

    X = np.fft.rfft(source_signal, n=padded_len)
    freqs = np.fft.rfftfreq(padded_len, 1.0/fs)

    for m in range(n_sensors):
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delays_sec[m])
        X_shifted = X * phase_shift
        y_shifted = np.fft.irfft(X_shifted, n=padded_len)
        output[m, :] = y_shifted * attenuation[m]

    # We return the padded length so that calling functions can handle
    # the delayed energy correctly (e.g. by overlapping with the next block)
    # instead of blindly truncating it.
    return output

def generate_diffuse_noise(mic_positions, fs, n_samples, low_freq_rumble=True):
    """
    Generates spatially diffuse noise representing wind/pseudosound using
    cylindrical isotropic coherence matrix: Gamma_jk(f) = J0(2*pi*f*d/c).
    """
    n_sensors = len(mic_positions)

    # White noise base
    noise = np.random.randn(n_sensors, n_samples)

    X = np.fft.rfft(noise, axis=1)
    freqs = np.fft.rfftfreq(n_samples, 1/fs)

    # Precompute distances between all mic pairs
    dists = np.zeros((n_sensors, n_sensors))
    for j in range(n_sensors):
        for k in range(n_sensors):
            dists[j, k] = np.linalg.norm(mic_positions[j] - mic_positions[k])

    # Apply shaping per frequency bin
    for idx, f in enumerate(freqs):
        if f == 0:
            continue

        # Isotropic coherence matrix
        arg = 2 * np.pi * f * dists / c
        gamma = sp.jv(0, arg) # Bessel function of first kind, order 0

        # Ensure positive semi-definite due to numerical issues
        eigenvalues, eigenvectors = np.linalg.eigh(gamma)
        eigenvalues[eigenvalues < 0] = 0
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # Correlate the independent noise
        X[:, idx] = L @ X[:, idx]

        # Apply -6dB/octave (1/f) low-frequency rumble
        if low_freq_rumble:
            X[:, idx] *= (1.0 / np.sqrt(f))

    diffuse = np.fft.irfft(X, n=n_samples, axis=1)

    # Normalize power
    diffuse /= np.std(diffuse)
    return diffuse

def apply_turbulence_decoherence(signal_matrix, fs, mic_positions, sigma_phi_base=0.1, distance=120.0):
    """
    Perturbs the phase of the signal to mimic thermal and wind eddies.
    sigma_phi(f, d) = alpha * f * sqrt(d)
    """
    if sigma_phi_base == 0.0:
        return signal_matrix

    n_sensors, n_samples = signal_matrix.shape
    X = np.fft.rfft(signal_matrix, axis=1)
    freqs = np.fft.rfftfreq(n_samples, 1/fs)

    for m in range(n_sensors):
        # Calculate standard deviation of phase jitter per frequency bin
        sigma_phi = sigma_phi_base * freqs * np.sqrt(distance) / 1000.0 # scaled

        # Generate random phase jitter
        delta_phi = np.random.normal(0, sigma_phi)

        # Apply phase perturbation
        X[m, :] *= np.exp(1j * delta_phi)

    return np.fft.irfft(X, n=n_samples, axis=1)

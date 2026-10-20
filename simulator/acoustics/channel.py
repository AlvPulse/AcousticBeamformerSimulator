import numpy as np
import scipy.special as sp

c = 343.0 # Speed of sound (m/s)

def calculate_iso_9613_alpha(f, temp_c=20.0, humidity_pct=50.0, p_atm=101.325):
    """
    Calculates atmospheric attenuation coefficient alpha (dB/m) according to ISO 9613-1.
    f: Frequency array in Hz.
    Returns: alpha in linear scale (multiplicative attenuation factor per meter).
    """
    # Prevent divide by zero for f=0
    f = np.maximum(f, 1e-10)

    # Reference atmospheric pressure (kPa)
    pr = 101.325
    pa = p_atm

    # Absolute temperature (K)
    T = temp_c + 273.15
    T01 = 293.15

    # Saturation vapor pressure
    C = -6.8346 * (T01 / T)**1.261 + 4.6151
    psat = pr * 10**C

    # Molar concentration of water vapor (%)
    h = humidity_pct * (psat / pa)

    # Relaxation frequencies for Oxygen and Nitrogen (Hz)
    frO = (pa / pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa / pr) * np.sqrt(T / T01) * (9 + 280 * h * np.exp(-4.170 * ((T / T01)**(-1/3) - 1)))

    # Attenuation coefficient alpha (dB / m)
    # Note: Formula from ISO 9613-1 gives dB/km, we divide by 1000 for dB/m
    alpha_db_km = 8.686 * f**2 * (
        (1.84e-11 * (pa/pr)**-1 * np.sqrt(T/T01)) +
        ((T/T01)**-2.5 * (0.01275 * np.exp(-2239.1 / T) * (frO / (frO**2 + f**2)) +
                          0.1068 * np.exp(-3352.0 / T) * (frN / (frN**2 + f**2))))
    )
    alpha_db_m = alpha_db_km / 1000.0

    # Convert dB/m to a linear multiplier per meter
    # dB = 20 * log10(A), so A = 10^(dB/20). Since it's attenuation, it's negative dB.
    alpha_linear = 10 ** (-alpha_db_m / 20.0)
    return alpha_linear

def apply_greens_function(source_signal, fs, source_pos, mic_positions, apply_iso=True):
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

        # Apply ISO 9613-1 atmospheric absorption filter if requested
        if apply_iso:
            iso_alpha = calculate_iso_9613_alpha(freqs)
            # Alpha is per meter. Total atmospheric attenuation is alpha^dist
            X_shifted = X_shifted * (iso_alpha ** distances[m])

        y_shifted = np.fft.irfft(X_shifted, n=padded_len)
        # Apply 1/R spherical spreading attenuation
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

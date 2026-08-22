import numpy as np

def array_factor(array, freq_hz, look_az, look_el, grid_az, grid_el, c=343.0, spatial_weights=None):
    """
    Stage 0: Computes the theoretical array factor (noiseless, no propagation).
    array factor = |Sum_i w_i * exp(j * 2*pi/lambda * (p_i dot (k_grid - k_look)))|^2

    Returns:
        af: 2D numpy array of shape (len(grid_el), len(grid_az))
    """
    wavelength = c / freq_hz
    k = 2 * np.pi / wavelength

    pos = array.positions # (n_sensors, 3)

    # Look vector (steering direction)
    look_az_rad = np.deg2rad(look_az)
    look_el_rad = np.deg2rad(look_el)
    k_look = np.array([
        np.cos(look_el_rad) * np.cos(look_az_rad),
        np.cos(look_el_rad) * np.sin(look_az_rad),
        np.sin(look_el_rad)
    ])

    # Grid vectors
    AZ, EL = np.meshgrid(np.deg2rad(grid_az), np.deg2rad(grid_el))
    k_grid_x = np.cos(EL) * np.cos(AZ)
    k_grid_y = np.cos(EL) * np.sin(AZ)
    k_grid_z = np.sin(EL)

    # Shape of grid is (n_el, n_az)
    af = np.zeros(AZ.shape, dtype=complex)

    if spatial_weights is None:
        spatial_weights = np.ones(array.n_sensors)

    for i in range(array.n_sensors):
        p_i = pos[i]
        w_i = spatial_weights[i]

        # p_i dot k_grid
        dot_grid = p_i[0]*k_grid_x + p_i[1]*k_grid_y + p_i[2]*k_grid_z
        # p_i dot k_look
        dot_look = np.dot(p_i, k_look)

        phase = k * (dot_grid - dot_look)
        af += w_i * np.exp(1j * phase)

    af = np.abs(af)**2
    # Normalize
    af /= np.max(af)

    return af

def generate_harmonic_weights(freqs, f0, n_harmonics, bandwidth_hz):
    """
    Generates a comb filter (spectral mask) for a known harmonic target (e.g., FPV or Siren).
    """
    mask = np.zeros_like(freqs)
    for h in range(1, n_harmonics + 1):
        center = h * f0
        lower = center - bandwidth_hz / 2
        upper = center + bandwidth_hz / 2
        mask[(freqs >= lower) & (freqs <= upper)] = 1.0
    return mask

def generate_broadband_envelope(freqs):
    """
    Generates a broadband 1/f spectral template typical of aircraft or distant machinery.
    """
    mask = np.zeros_like(freqs)
    valid = freqs > 0
    mask[valid] = 1.0 / freqs[valid]
    # Normalize mask
    mask = mask / np.max(mask)
    return mask

def delay_and_sum(recording, array, fs, grid_az, grid_el, c=343.0, spatial_weights=None, freq_weights=None):
    """
    Stage 1: Conventional Delay-And-Sum (Bartlett) empirical beamformer.
    Steered with a far-field plane-wave assumption.

    recording: (n_sensors, n_samples)
    freq_weights: Optional 1D array of length (NFFT/2), acts as a Spectral Matched Filter before power summation.
    """
    n_sensors, n_samples = recording.shape
    pos = array.positions

    NFFT = n_samples
    X = np.fft.rfft(recording, n=NFFT, axis=1) # (n_sensors, n_bins)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)

    valid_bins = freqs > 0
    X = X[:, valid_bins]
    freqs = freqs[valid_bins]

    if freq_weights is not None:
        # Align weights to valid bins
        freq_weights = freq_weights[valid_bins]
    else:
        freq_weights = np.ones_like(freqs)

    AZ, EL = np.meshgrid(np.deg2rad(grid_az), np.deg2rad(grid_el))
    k_grid_x = np.cos(EL) * np.cos(AZ)
    k_grid_y = np.cos(EL) * np.sin(AZ)
    k_grid_z = np.sin(EL)

    power_map = np.zeros(AZ.shape)

    if spatial_weights is None:
        spatial_weights = np.ones(n_sensors)

    n_el, n_az = AZ.shape

    for r in range(n_el):
        for c_idx in range(n_az):
            kx = k_grid_x[r, c_idx]
            ky = k_grid_y[r, c_idx]
            kz = k_grid_z[r, c_idx]
            k_vec = np.array([kx, ky, kz])

            tau = -np.dot(pos, k_vec) / c
            phase = 2 * np.pi * np.outer(tau, freqs)

            steered_X = X * np.exp(1j * phase)
            steered_X = steered_X * spatial_weights[:, np.newaxis]

            Y = np.sum(steered_X, axis=0)

            # Frequency-weighted Power Summation (Matched Filter)
            weighted_power = np.abs(Y)**2 * freq_weights
            power = np.sum(weighted_power)

            power_map[r, c_idx] = power

    if np.max(power_map) > 0:
        power_map /= np.max(power_map)

    return power_map

def compute_phase_stor(recording, array, fs, target_az, target_el, c=343.0, freq_weights=None):
    """
    Computes Phase-STOR (SRP-PHAT style).
    Normalizes each frequency bin by its magnitude before steering,
    making the metric purely phase-dependent. Highly resistant to
    loud amplitude noise (like wind gusts).
    """
    n_sensors, n_samples = recording.shape
    pos = array.positions

    NFFT = n_samples
    X = np.fft.rfft(recording, n=NFFT, axis=1)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)

    valid_bins = freqs > 0
    X = X[:, valid_bins]
    freqs = freqs[valid_bins]

    # Apply PHAT (Phase Transform) normalization
    magnitudes = np.abs(X)
    # Avoid division by zero
    magnitudes[magnitudes < 1e-10] = 1e-10
    X_phat = X / magnitudes

    if freq_weights is not None:
        freq_weights = freq_weights[valid_bins]
    else:
        freq_weights = np.ones_like(freqs)

    k_vec = np.array([
        np.cos(np.deg2rad(target_el)) * np.cos(np.deg2rad(target_az)),
        np.cos(np.deg2rad(target_el)) * np.sin(np.deg2rad(target_az)),
        np.sin(np.deg2rad(target_el))
    ])

    tau = -np.dot(pos, k_vec) / c
    phase = 2 * np.pi * np.outer(tau, freqs)

    steered_X = X_phat * np.exp(1j * phase)
    # Average the signals coherently FIRST
    Y = np.mean(steered_X, axis=0)

    steered_power = np.sum(np.abs(Y)**2 * freq_weights)

    omni_power = np.mean([np.sum(np.abs(X_phat[i])**2 * freq_weights) for i in range(n_sensors)])

    if omni_power <= 0:
        return 0.0

    return 10 * np.log10(steered_power / omni_power)

def compute_stor(recording, array, fs, target_az, target_el, c=343.0, freq_weights=None):
    """
    Computes Steered-To-Omni Ratio (STOR).
    A low-compute O(1) alternative to PAPR/PMPR.
    Divides the power steered directly at the target by the average omni-directional power of the mics.
    If the array successfully aligns the signal, this ratio spikes.
    """
    n_sensors, n_samples = recording.shape
    pos = array.positions

    NFFT = n_samples
    X = np.fft.rfft(recording, n=NFFT, axis=1)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)

    valid_bins = freqs > 0
    X = X[:, valid_bins]
    freqs = freqs[valid_bins]

    if freq_weights is not None:
        freq_weights = freq_weights[valid_bins]
    else:
        freq_weights = np.ones_like(freqs)

    k_vec = np.array([
        np.cos(np.deg2rad(target_el)) * np.cos(np.deg2rad(target_az)),
        np.cos(np.deg2rad(target_el)) * np.sin(np.deg2rad(target_az)),
        np.sin(np.deg2rad(target_el))
    ])

    tau = -np.dot(pos, k_vec) / c
    phase = 2 * np.pi * np.outer(tau, freqs)

    # Steered Power
    steered_X = X * np.exp(1j * phase)
    # Average the signals coherently FIRST, then compute power
    # This prevents artificial gain from just summing uncorrelated noise
    Y = np.mean(steered_X, axis=0)

    steered_power = np.sum(np.abs(Y)**2 * freq_weights)

    # Omni Power (Average power of individual un-steered microphones)
    # We sum their powers and average them
    omni_power = np.mean([np.sum(np.abs(X[i])**2 * freq_weights) for i in range(n_sensors)])

    if omni_power <= 0:
        return 0.0

    return 10 * np.log10(steered_power / omni_power)

def compute_map_papr(power_map_linear):
    """
    Computes Peak-to-Average Power Ratio (PAPR) of the 2D power map in dB.
    High PAPR = Sharp distinct peak. Low PAPR = Noise / Ambiguity everywhere.
    """
    peak = np.max(power_map_linear)
    avg = np.mean(power_map_linear)
    if avg <= 0:
        return 0.0
    return 10 * np.log10(peak / avg)

def compute_isl(power_map_linear, grid_az, grid_el, true_az, true_el, mainlobe_radius_deg=15.0):
    """
    Computes Integrated Sidelobe Level (ISL) in dB.
    Ratio of total energy outside the mainlobe to energy inside the mainlobe.
    """
    AZ, EL = np.meshgrid(grid_az, grid_el)

    # Distance from true DOA
    # Approximate angular distance for a flat map (valid for narrow mainlobes)
    d_az = AZ - true_az
    # wrap az distance to [-180, 180]
    d_az = (d_az + 180) % 360 - 180
    d_el = EL - true_el

    dist_sq = d_az**2 + d_el**2
    radius_sq = mainlobe_radius_deg**2

    mainlobe_mask = dist_sq <= radius_sq

    main_energy = np.sum(power_map_linear[mainlobe_mask])
    sidelobe_energy = np.sum(power_map_linear[~mainlobe_mask])

    if main_energy <= 0 or sidelobe_energy <= 0:
        return -np.inf # No sidelobes or no mainlobe

    return 10 * np.log10(sidelobe_energy / main_energy)

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

def delay_and_sum(recording, array, fs, grid_az, grid_el, c=343.0, spatial_weights=None):
    """
    Stage 1: Conventional Delay-And-Sum (Bartlett) empirical beamformer.
    Steered with a far-field plane-wave assumption.

    recording: (n_sensors, n_samples)
    """
    n_sensors, n_samples = recording.shape
    pos = array.positions

    # FFT of recording for frequency domain shifting (faster for grid search than time-domain interpolation)
    # Actually, we can do it in frequency domain for all frequencies at once, or narrow band.
    # The prompt implies a broad-band or narrow-band generic DAS.
    # A standard time-domain DAS:
    # y(t) = Sum_i w_i * x_i(t - tau_i)
    # Power = mean(y(t)^2)
    # Doing this in time domain with interpolation for every grid point is slow.
    # Doing it in frequency domain:
    # Y(f) = Sum_i w_i * X_i(f) * exp(-j * 2*pi*f * tau_i)
    # Power = Sum_f |Y(f)|^2

    # Let's use the Frequency Domain approach
    NFFT = n_samples
    X = np.fft.rfft(recording, n=NFFT, axis=1) # (n_sensors, n_bins)
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)

    # Ignore DC
    valid_bins = freqs > 0
    X = X[:, valid_bins]
    freqs = freqs[valid_bins]

    AZ, EL = np.meshgrid(np.deg2rad(grid_az), np.deg2rad(grid_el))
    k_grid_x = np.cos(EL) * np.cos(AZ)
    k_grid_y = np.cos(EL) * np.sin(AZ)
    k_grid_z = np.sin(EL)

    power_map = np.zeros(AZ.shape)

    if spatial_weights is None:
        spatial_weights = np.ones(n_sensors)

    # To optimize memory, we iterate over grid points
    n_el, n_az = AZ.shape

    for r in range(n_el):
        for c_idx in range(n_az):
            # Steering vector direction (towards grid point)
            kx = k_grid_x[r, c_idx]
            ky = k_grid_y[r, c_idx]
            kz = k_grid_z[r, c_idx]
            k_vec = np.array([kx, ky, kz])

            # tau_i = - (p_i dot k_vec) / c
            # We want to align the signals. If a signal comes from k_vec, it has delay (p_i dot k_vec) / c
            # To compensate, we apply delay tau_i = - (p_i dot k_vec) / c
            tau = -np.dot(pos, k_vec) / c

            # Beamformed signal in frequency domain
            # Y(f) = Sum_i w_i * X_i(f) * exp(j * 2*pi*f * tau_i)
            # (Note sign: if signal arrived with phase -w*tau, we multiply by +w*tau to align)

            # tau is shape (n_sensors,)
            # freqs is shape (n_bins,)
            # phase is (n_sensors, n_bins)
            phase = 2 * np.pi * np.outer(tau, freqs)

            steered_X = X * np.exp(1j * phase)
            # Apply weights
            steered_X = steered_X * spatial_weights[:, np.newaxis]

            # Sum over sensors
            Y = np.sum(steered_X, axis=0)

            # Power
            power = np.sum(np.abs(Y)**2)
            power_map[r, c_idx] = power

    # Normalize
    power_map /= np.max(power_map)

    return power_map

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

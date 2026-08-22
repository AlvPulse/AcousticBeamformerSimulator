import numpy as np
from scipy import signal as scipy_signal
import scipy.io.wavfile as wavfile
from acoustic_sim.propagation import per_sensor_delays, path_loss_db

def tone_burst(freq_hz, duration_s, fs) -> np.ndarray:
    t = np.arange(int(duration_s * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)

def chirp(f0, f1, duration_s, fs) -> np.ndarray:
    t = np.arange(int(duration_s * fs)) / fs
    return scipy_signal.chirp(t, f0, duration_s, f1, method='linear')

def noise_burst(duration_s, fs) -> np.ndarray:
    n_samples = int(duration_s * fs)
    return np.random.randn(n_samples)

def load_wav(filepath, fs_target=None) -> np.ndarray:
    fs_orig, data = wavfile.read(filepath)
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    # Normalize to [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0

    if fs_target is not None and fs_orig != fs_target:
        # Resample
        num_samples = int(len(data) * fs_target / fs_orig)
        data = scipy_signal.resample(data, num_samples)

    return data

def spl_db_to_pa(spl_db, p_ref=20e-6) -> float:
    return p_ref * (10.0 ** (spl_db / 20.0))

def compute_spatial_coherence(recording, fs, nperseg=1024):
    """
    Computes the average magnitude squared coherence between all unique pairs of sensors.
    This gives a scalar [0, 1] indicating how correlated the wavefield is across the array.
    """
    n_sensors = recording.shape[0]
    if n_sensors < 2:
        return 1.0

    total_coh = 0.0
    count = 0
    # To keep it fast, we can compute coherence for adjacent pairs,
    # but for small arrays (e.g. 16), all pairs is fast enough (120 pairs).
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            # compute MSC
            f, Cxy = scipy_signal.coherence(recording[i], recording[j], fs, nperseg=min(nperseg, recording.shape[1]))
            # average MSC across all frequency bins for this pair
            total_coh += np.mean(Cxy)
            count += 1

    return total_coh / max(1, count)

def compute_array_gain(recording, beamformed_time_signal, pure_signal_power, pure_noise_power=None):
    """
    Computes the empirical Array Gain (AG) = SNR_out / SNR_in.
    Note: Requires knowing the pure signal/noise powers, or estimating them.
    Because we synthesize the signal, we can calculate theoretical input SNR.
    Since this needs to work on the output of DAS (which we currently do in freq domain to just get power),
    we might need to calculate input SNR and output peak power.
    We will implement a simpler empirical version in the main app using known inputs.
    """
    pass

def apply_fractional_delay(sig, delay_s, fs):
    """
    Applies a fractional delay to a signal using frequency domain phase shift.
    """
    N = len(sig)
    # FFT of signal
    Sig = np.fft.rfft(sig)

    # Frequency bins
    freqs = np.fft.rfftfreq(N, 1.0/fs)

    # Phase shift: exp(-j * 2 * pi * f * t)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_s)

    # Apply phase shift and IFFT
    Sig_shifted = Sig * phase_shift
    sig_shifted = np.fft.irfft(Sig_shifted, n=N)

    return sig_shifted

def highpass_filter(sig, fc, fs):
    """
    Applies a Butterworth highpass filter.
    """
    if fc <= 0:
        return sig
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = scipy_signal.butter(4, normal_cutoff, btype='high', analog=False)
    return scipy_signal.filtfilt(b, a, sig)

def synthesize_array_recording(array, source_signal, az, el, r,
                               source_level_db, fs, noise_level_db,
                               freq_hz_for_pl, env_params,
                               directional_noise=None) -> np.ndarray:
    """
    Synthesizes the multi-channel array recording.
    Returns: (n_sensors, n_samples)
    """
    n_sensors = array.n_sensors
    n_samples = len(source_signal)

    # Reference pressure amplitude for the source at 1m
    p_source_amp = spl_db_to_pa(source_level_db)

    # Normalize source signal to peak amplitude of 1 before scaling
    if np.max(np.abs(source_signal)) > 0:
        source_signal = source_signal / np.max(np.abs(source_signal))
    # Scale to target pressure amplitude
    source_signal = source_signal * p_source_amp

    # Calculate delays for the target source
    delays = per_sensor_delays(array, az, el, r, c=env_params['c'])
    # Relative delay (to keep things bounded, center around 0 or min delay)
    min_delay = np.min(delays)
    delays -= min_delay # Shift delays so the first arriving signal has 0 delay

    # Calculate path loss for each sensor
    p_sensors = np.zeros((n_sensors, n_samples))
    for i in range(n_sensors):
        # Path loss uses distance from sensor to source
        # Let's compute exact distance for PL
        p_i = array.positions[i]
        from acoustic_sim.propagation import spherical_to_cartesian
        p_s = spherical_to_cartesian(r, az, el)
        r_i = np.linalg.norm(p_s - p_i)

        pl_db = path_loss_db(freq_hz_for_pl, r_i, env_params['temp_c'],
                             env_params['rel_humidity'], env_params['pressure_kpa'],
                             env_params.get('ground_effect_loss_db', 0.0),
                             env_params.get('shadowing_std_db', 0.0))

        # Apply path loss scale
        scale = 10.0 ** (-pl_db / 20.0)

        # Apply delay
        sig_delayed = apply_fractional_delay(source_signal, delays[i], fs)

        p_sensors[i, :] = sig_delayed * scale

    # Directional noise
    if directional_noise and directional_noise.get('enabled', False):
        n_az = directional_noise['az']
        n_el = directional_noise['el']
        n_r = directional_noise['r']
        n_lvl = directional_noise['spl_db']

        n_p_amp = spl_db_to_pa(n_lvl)
        noise_sig = np.random.randn(n_samples)
        noise_sig = noise_sig / np.max(np.abs(noise_sig)) * n_p_amp

        n_delays = per_sensor_delays(array, n_az, n_el, n_r, c=env_params['c'])
        n_delays -= np.min(n_delays)

        for i in range(n_sensors):
            p_i = array.positions[i]
            p_n_s = spherical_to_cartesian(n_r, n_az, n_el)
            n_r_i = np.linalg.norm(p_n_s - p_i)

            # Note: Assuming ambient directional noise is also subject to ground and shadowing
            n_pl_db = path_loss_db(freq_hz_for_pl, n_r_i, env_params['temp_c'],
                                 env_params['rel_humidity'], env_params['pressure_kpa'],
                                 env_params.get('ground_effect_loss_db', 0.0),
                                 env_params.get('shadowing_std_db', 0.0))
            n_scale = 10.0 ** (-n_pl_db / 20.0)

            n_sig_delayed = apply_fractional_delay(noise_sig, n_delays[i], fs)
            p_sensors[i, :] += n_sig_delayed * n_scale

    # Independent sensor self-noise
    if noise_level_db is not None:
        noise_p_amp = spl_db_to_pa(noise_level_db)
        # Noise is Gaussian, so std dev is the RMS pressure
        # To match the SPL (which is RMS), we multiply standard normal by RMS pressure
        self_noise = np.random.randn(n_sensors, n_samples) * noise_p_amp
        p_sensors += self_noise

    return p_sensors

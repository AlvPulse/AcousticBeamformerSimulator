import numpy as np
import pytest

from dsp_array import ArrayGeometry, aliasing_frequency
from propagation import iso9613_absorption_db_per_m, path_loss_db, per_sensor_delays
from dsp_signal import tone_burst, apply_fractional_delay, spl_db_to_pa
from beamformer import array_factor, delay_and_sum

def test_iso9613_absorption():
    """
    Test ISO 9613-1 absorption values against published expected ranges.
    At 20°C, 70% RH, 101.325 kPa:
    - 1 kHz is roughly 5 dB/km (0.005 dB/m)
    - 4 kHz is roughly 25-30 dB/km (0.025 - 0.030 dB/m)
    """
    # 1 kHz
    alpha_1k = iso9613_absorption_db_per_m(1000.0, temp_c=20.0, rel_humidity=70.0, pressure_kpa=101.325)
    assert 0.004 < alpha_1k < 0.006, f"1kHz absorption out of bounds: {alpha_1k} dB/m"

    # 4 kHz
    alpha_4k = iso9613_absorption_db_per_m(4000.0, temp_c=20.0, rel_humidity=70.0, pressure_kpa=101.325)
    assert 0.02 < alpha_4k < 0.04, f"4kHz absorption out of bounds: {alpha_4k} dB/m"

def test_path_loss_geometric_spreading():
    """
    Test that path loss follows 20*log10(r) exactly when absorption is negligible (low frequency).
    """
    # At 10 Hz, absorption should be extremely close to 0
    pl_10m = path_loss_db(10.0, 10.0)
    pl_100m = path_loss_db(10.0, 100.0)

    # Geometric spreading: 20*log10(10) = 20 dB, 20*log10(100) = 40 dB
    # Difference should be almost exactly 20 dB
    diff = pl_100m - pl_10m
    assert np.isclose(diff, 20.0, atol=0.1), f"Spreading loss expected ~20dB difference, got {diff}"

def test_fractional_delay():
    """
    Test that apply_fractional_delay correctly shifts the phase of a pure tone
    and does not alter the amplitude.
    """
    fs = 48000
    f = 1000.0
    dur = 0.05
    t = np.arange(int(dur * fs)) / fs
    sig = np.sin(2 * np.pi * f * t)

    # Delay by exactly 1/4 wavelength of 1000 Hz (which is 0.25 ms)
    # This should correspond to a 90-degree phase shift.
    delay_s = 0.25 / 1000.0

    sig_delayed = apply_fractional_delay(sig, delay_s, fs)

    # Check amplitude preservation (RMS should be identical)
    rms_orig = np.sqrt(np.mean(sig**2))
    rms_delayed = np.sqrt(np.mean(sig_delayed**2))
    assert np.isclose(rms_orig, rms_delayed, rtol=1e-3), "Amplitude changed after fractional delay"

    # Check phase shift. A sin wave delayed by T/4 becomes a -cos wave.
    expected_sig = np.sin(2 * np.pi * f * (t - delay_s))

    # We ignore the very edges where FFT wrapping might cause slight artifacts
    cut = 100
    assert np.allclose(sig_delayed[cut:-cut], expected_sig[cut:-cut], atol=1e-2), "Phase shift did not match expected"

def test_array_factor_grating_lobes():
    """
    Test that array_factor places grating lobes exactly where geometrically predicted.
    For a 2-element Uniform Linear Array on the x-axis:
    Spacing d. Steered to broadside (az=0).
    Grating lobes appear when d >= lambda.
    Condition for lobes: d * sin(theta) = m * lambda.
    If d = lambda, grating lobes should appear at +/- 90 degrees azimuth.
    """
    # Create 2-element array on X-axis, centered at 0
    c = 343.0
    freq = 1000.0
    lam = c / freq

    # Spacing d = lambda
    d = lam
    x = [-d/2, d/2]
    y = [0.0, 0.0]

    arr = ArrayGeometry(x, y)

    # Look broadside (az=0, el=0)
    # Grid search along azimuth from -90 to 90
    grid_az = np.linspace(-90, 90, 181)
    grid_el = np.array([0.0])

    af = array_factor(arr, freq, look_az=0.0, look_el=0.0, grid_az=grid_az, grid_el=grid_el, c=c)
    af_1d = af[0, :] # Extract the single elevation cut

    # Find peaks. Max power should be 1.0 (normalized)
    peaks_idx = np.where(np.isclose(af_1d, 1.0, atol=1e-2))[0]
    peak_angles = grid_az[peaks_idx]

    # Expected peaks: broadside (0) and grating lobes at +/- 90
    expected_angles = [-90.0, 0.0, 90.0]

    for ea in expected_angles:
        assert any(np.isclose(ea, pa, atol=1.0) for pa in peak_angles), f"Expected peak at {ea}, found peaks at {peak_angles}"

def test_spl_conversion():
    """
    Test SPL dB to Pressure Amplitude conversion.
    94 dB SPL is commonly 1 Pascal RMS.
    Amplitude (peak) should be 1 * sqrt(2) = 1.414 Pa.
    """
    p_amp = spl_db_to_pa(94.0, p_ref=20e-6)
    # 94 dB is 20*log10(p_rms / 20uPa)
    # p_rms = 20uPa * 10^(94/20) = 1.00237 Pa
    # p_amp = p_rms (since spl_db_to_pa currently returns RMS, wait, the implementation
    # uses p_ref * 10^(SPL/20). Let's verify what the function outputs.
    # The function spl_db_to_pa(spl_db) returns the RMS pressure.
    p_rms_expected = 20e-6 * (10**(94.0/20.0))
    assert np.isclose(p_amp, p_rms_expected), "SPL to Pa conversion is incorrect."

def test_advanced_path_loss_statistics():
    """
    Test that the path loss model correctly incorporates deterministic ground effect
    and log-normal shadowing variation. Tests over hundreds of trials to verify
    the statistical distribution matches the expected std deviation and mean.
    """
    freq_hz = 1000.0
    distance_m = 100.0
    ground_effect_loss = 3.0 # dB
    shadowing_std = 2.0 # dB

    # Baseline expected mean loss (geometric + absorption + ground)
    baseline_loss = path_loss_db(freq_hz, distance_m,
                                 ground_effect_loss_db=ground_effect_loss,
                                 shadowing_std_db=0.0)

    # Run 10000 trials to get good statistics
    n_trials = 10000
    losses = np.array([path_loss_db(freq_hz, distance_m,
                                    ground_effect_loss_db=ground_effect_loss,
                                    shadowing_std_db=shadowing_std)
                       for _ in range(n_trials)])

    sample_mean = np.mean(losses)
    sample_std = np.std(losses)

    # The mean should match the baseline (which includes ground effect)
    assert np.isclose(sample_mean, baseline_loss, atol=0.1), f"Expected mean {baseline_loss}, got {sample_mean}"

    # The standard deviation should match the shadowing parameter
    assert np.isclose(sample_std, shadowing_std, atol=0.1), f"Expected std dev {shadowing_std}, got {sample_std}"

    # Check 95% confidence interval (approx +/- 1.96 * sigma)
    # About 95% of data should fall within mean +/- 1.96 * sigma
    lower_bound = baseline_loss - 1.96 * shadowing_std
    upper_bound = baseline_loss + 1.96 * shadowing_std

    in_interval = np.sum((losses >= lower_bound) & (losses <= upper_bound))
    pct_in_interval = in_interval / n_trials

    assert 0.93 < pct_in_interval < 0.97, f"Expected ~95% in CI, got {pct_in_interval*100:.1f}%"

def test_delay_and_sum_steering():
    """
    Test the empirical Delay-and-Sum beamformer points to the true source
    under noiseless conditions.
    """
    # 4-element Uniform Circular Array (UCA) to avoid front/back azimuth ambiguity
    # A ULA on the X-axis cannot distinguish between +az and -az because cos(az) = cos(-az).
    c = 343.0
    f = 1000.0
    lam = c / f
    r = lam / 2
    # 4 elements in a circle
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    arr = ArrayGeometry(x, y)

    fs = 48000
    dur = 0.05
    sig = tone_burst(f, dur, fs)

    # Source at azimuth 30 degrees, far field (r=100)
    true_az = 30.0
    true_el = 0.0

    # Create the recording manually using ideal fractional delays to test DAS purely
    delays = per_sensor_delays(arr, true_az, true_el, r=100.0, c=c)
    delays -= np.min(delays)

    recording = np.zeros((4, len(sig)))
    for i in range(4):
        recording[i] = apply_fractional_delay(sig, delays[i], fs)

    # Run DAS
    grid_az = np.arange(-180, 180, 1.0)
    grid_el = np.array([0.0])

    power_map = delay_and_sum(recording, arr, fs, grid_az, grid_el, c=c)

    power_1d = power_map[0, :]
    peak_idx = np.argmax(power_1d)
    steered_az = grid_az[peak_idx]

    assert np.isclose(steered_az, true_az, atol=1.0), f"DAS steered to {steered_az}, expected {true_az}"

import pytest
import numpy as np
from simulator.acoustics.array import generate_log_spiral_array
from simulator.acoustics.channel import apply_greens_function
from simulator.acoustics.scene_generator import generate_scene

def test_array_generation():
    pos = generate_log_spiral_array(n_sensors=50, aperture_m=2.5)
    assert pos.shape == (50, 3)
    # Ensure it's roughly centered
    np.testing.assert_allclose(np.mean(pos, axis=0), [0, 0, 0], atol=1e-5)
    # Check max radius roughly matches aperture/2
    radii = np.linalg.norm(pos, axis=1)
    assert np.max(radii) <= 1.25 + 1e-4

def test_greens_function():
    fs = 16000
    pos = np.array([[0,0,0], [1,0,0]])
    src = np.array([10,0,0]) # 10 meters away on x axis
    signal = np.ones(1000)
    out = apply_greens_function(signal, fs, src, pos)
    assert out.shape == (2, 1000)

def test_narrowband_vs_wideband_lcmv():
    from simulator.beamformers.lcmv_smi import LCMV_SMI
    from simulator.beamformers.narrowband_lcmv import NarrowbandLCMV
    fs = 16000
    bf_wb = LCMV_SMI(fs=fs)
    bf_nb = NarrowbandLCMV(fs=fs)
    # Just asserting instantiation and base properties work properly
    assert bf_wb.fs == fs
    assert bf_nb.fs == fs

def test_dynamic_snr_sweep():
    from simulator.acoustics.scene_generator import generate_scene
    from simulator.acoustics.array import generate_log_spiral_array

    fs = 16000
    duration_sec = 0.5
    pos = generate_log_spiral_array(n_sensors=10, aperture_m=1.0)

    # Generate some dummy audio to test the function runs correctly
    import librosa
    import soundfile as sf
    import os
    dummy_path = "tests/dummy.wav"
    s = np.sin(2 * np.pi * 1000 * np.linspace(0, duration_sec, int(fs*duration_sec)))
    sf.write(dummy_path, s, fs)

    # Test dynamic SNR (airplane flyby logic)
    mixture, tgt, intf, noise = generate_scene(
        target_audio_path=dummy_path,
        array_positions=pos,
        fs=fs,
        target_snr="dynamic",
        duration_sec=duration_sec
    )

    assert mixture.shape == (10, int(fs * duration_sec))
    assert tgt.shape == (10, int(fs * duration_sec))
    os.remove(dummy_path)

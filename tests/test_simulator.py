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

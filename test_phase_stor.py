import numpy as np
import scipy.signal as signal
from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.beamformer import compute_phase_stor

pos = ArrayGeometry(np.zeros(50), np.zeros(50))
audio = np.random.randn(50, 16000)
grid_az = np.linspace(-180, 180, 8)
grid_el = np.linspace(0, 90, 4)
fs = 16000

try:
    p_map = compute_phase_stor(audio, pos, fs, grid_az, grid_el)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()

from experiments.evaluate_real_world import compute_phase_stor_metric
import numpy as np
from acoustic_sim.dsp_array import ArrayGeometry

fs = 16000
pos = ArrayGeometry(np.zeros(50), np.zeros(50))
audio = np.random.randn(50, fs)
target_az = 0.0
target_el = 0.0

res = compute_phase_stor_metric(audio, pos, target_az, target_el, fs)
print("Metric:", res)

import numpy as np
from simulator.beamformers.das import StandardDAS
from simulator.beamformers.robust_socp import RobustSOCP
from acoustic_sim.dsp_array import ArrayGeometry

fs = 16000
pos = ArrayGeometry(np.zeros(50), np.zeros(50))
pos_3d = np.stack([pos.x, pos.y, np.zeros_like(pos.x)], axis=1)

audio = np.random.randn(50, 16000)

look_pos = np.array([1.0, 0.0, 0.0]) * 120.0

das = StandardDAS(pos_3d, fs)
socp = RobustSOCP(pos_3d, fs)

y_das = das.process(audio, look_pos)
y_socp = socp.process(audio, look_pos)

print("Power DAS:", np.mean(y_das**2))
print("Power SOCP:", np.mean(y_socp**2))

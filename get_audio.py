import librosa
import numpy as np
from acoustic_sim.dsp_array import load_array
from simulator.beamformers.lcmv_smi import LCMVSMI

audio, fs = librosa.load("data/field_tests/test_001/1.wav", sr=None, mono=False)
pos = load_array("array_geo.txt").positions
pos_3d = np.stack([pos[:,0], pos[:,1], pos[:,2]], axis=1)

lcmv = LCMVSMI(pos_3d, fs)

look_pos = np.array([120, 0, 0])
y_out = lcmv.process(audio, look_pos)

print("y_out:", np.mean(y_out**2))

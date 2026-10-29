import numpy as np
import scipy.signal as signal
from simulator.beamformers.das import StandardDAS
from acoustic_sim.dsp_array import ArrayGeometry
from simulator.beamformers.tapered_das import TaperedDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.cgmm_mvdr import cGMM_MVDR
from simulator.beamformers.robust_socp import RobustSOCP
from simulator.beamformers.ebb import EBB
from simulator.beamformers.multirank_osp import MultiRankOSP
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.evaluate_real_world import compute_phase_stor_metric

fs = 16000
pos = ArrayGeometry(np.zeros(50), np.zeros(50))
pos_3d = np.stack([pos.x, pos.y, np.zeros_like(pos.x)], axis=1)

audio = np.random.randn(50, 16000)

look_pos = np.array([1.0, 0.0, 0.0]) * 120.0

das = StandardDAS(pos_3d, fs)
print("das", np.mean(das.process(audio, look_pos)**2))

print("p_stor", compute_phase_stor_metric(audio, pos, 0.0, 0.0, fs))

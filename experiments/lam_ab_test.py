"""
Real LAM (AST audio-tagging) A/B test - the user's LAM-response question.

Scores the SAME scene's target through:
  1. clean siren reference
  2. DAS at target az (flat but contaminated)
  3. GSC at target az under +/-20 deg phase error (flat + clean)
  4. LCMV at target az under +/-20 deg phase error (colored: 2-3 dB
     line-gain spread -> does the classifier notice?)
  5. LCMV calibrated (best case LCMV)
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig
from transformers import pipeline

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_phase_robustness import (
    make_scene, apply_phase_err, AZ_T,
)

FS = 8000


def to16k(x):
    import librosa
    return librosa.resample(np.asarray(x).flatten(), orig_sr=FS, target_sr=16000)


def top_scores(clf, x, k=3):
    res = clf({"raw": to16k(x), "sampling_rate": 16000}, top_k=k)
    return ", ".join(f"{r['label']}={r['score']:.3f}" for r in res)


def main():
    print("loading AST (first run downloads ~350 MB)...", flush=True)
    clf = pipeline("audio-classification",
                   model="MIT/ast-finetuned-audioset-10-10-0.4593")
    print("AST ready", flush=True)

    mix, ref = make_scene(4000)
    rng = np.random.default_rng(4050)
    mix20 = apply_phase_err(mix, rng, 20.0)

    das = StandardDAS(POS if False else __import__(
        "experiments.benchmark_extreme_inr", fromlist=["POS"]).POS, FS)
    from experiments.benchmark_extreme_inr import POS, far_pos
    das = StandardDAS(POS, FS)
    lcmv = LCMVSMI(POS, FS)
    gsc = GSC_MCWF(POS, FS)

    cases = [
        ("clean siren reference", ref),
        ("DAS @target (20deg phase err)", das.process(mix20, far_pos(AZ_T))),
        ("GSC @target (20deg phase err)", gsc.process(mix20, far_pos(AZ_T))),
        ("LCMV @target calibrated", lcmv.process(mix, far_pos(AZ_T))),
        ("LCMV @target (20deg phase err)", lcmv.process(mix20, far_pos(AZ_T))),
    ]
    for name, y in cases:
        print(f"{name:32s} -> {top_scores(clf, y)}", flush=True)


if __name__ == "__main__":
    main()

"""
Dual-LAM cross-check: two AST variants (unbalanced + balanced training)
extractions, so 'what was the audio' is judged by two independent models.

Cases: clean target, GSC@phase-err, LCMV@phase-err, LCMV calibrated, DAS.
Target classes: tonal siren + machinery (chirp in extension).
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import librosa
from transformers import pipeline
from experiments.benchmark_phase_robustness import make_scene, apply_phase_err, AZ_T
from experiments.benchmark_extreme_inr import POS, FS, far_pos
from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.gsc_mcwf import GSC_MCWF


def to16k(x):
    return librosa.resample(np.asarray(x).flatten(), orig_sr=FS, target_sr=16000)


def main():
    print("loading AST...", flush=True)
    ast = pipeline("audio-classification",
                   model="MIT/ast-finetuned-audioset-10-10-0.4593")
    # second opinion: AST trained on the BALANCED Audioset variant
    # (different training balancing = genuinely different opinions).
    # (PANNs CNN14 blocked: zenodo checkpoint download unavailable here.)
    print("loading AST-0.4593 (balanced variant)...", flush=True)
    pan_ast = pipeline("audio-classification",
                       model="MIT/ast-finetuned-audioset-12-12-0.447")
    print("both LAMs ready", flush=True)

    mix, ref = make_scene(4000)
    rng = np.random.default_rng(4050)
    mix20 = apply_phase_err(mix, rng, 20.0)
    das, lcmv, gsc = StandardDAS(POS, FS), LCMVSMI(POS, FS), GSC_MCWF(POS, FS)

    cases = [
        ("clean_target", ref),
        ("GSC_phase20", gsc.process(mix20, far_pos(AZ_T))),
        ("LCMV_phase20", lcmv.process(mix20, far_pos(AZ_T))),
        ("LCMV_cal", lcmv.process(mix, far_pos(AZ_T))),
        ("DAS_phase20", das.process(mix20, far_pos(AZ_T))),
    ]
    print(f"\n{'case':14s} | {'AST top-2':46s} | {'PANNs top-2':46s}")
    print("-" * 112)
    for name, y in cases:
        x16 = to16k(y)[None, :]
        a = ast({"raw": to16k(y), "sampling_rate": 16000}, top_k=2)
        ast_s = ", ".join(f"{r['label'][:22]}={r['score']:.2f}" for r in a)
        b = pan_ast({"raw": to16k(y), "sampling_rate": 16000}, top_k=2)
        pan_s = ", ".join(f"{r['label'][:22]}={r['score']:.2f}" for r in b)
        print(f"{name:14s} | {ast_s:46s} | {pan_s}", flush=True)


if __name__ == "__main__":
    main()

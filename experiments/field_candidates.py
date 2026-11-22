"""
Real-world data check: extract candidate audio at the persistent field
directions with each method, classify with AST (single model, per user),
and score tonality. First 'methods on the real recordings' evidence.

Directions from the field quick pass (52-chunk persistence): ~8, ~22,
~-52, ~172 deg. Chunks sampled across the session: 1, 20, 40.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig
import soundfile as sf
import librosa
from transformers import pipeline

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_extreme_inr import POS, FS as FS8, far_pos, subspace_null
from experiments.benchmark_all_pipelines import tonality8k, gain_normalize

DIRS = [8.0, 22.0, -52.0, 172.0]
CHUNKS = ["1.wav", "20.wav", "40.wav"]


def main():
    print("loading AST (cached 10-10 model)...", flush=True)
    ast = pipeline("audio-classification",
                   model="MIT/ast-finetuned-audioset-10-10-0.4593")

    das, lcmv, gsc = StandardDAS(POS, FS8), LCMVSMI(POS, FS8), GSC_MCWF(POS, FS8)
    rows, outdir = [], os.path.join(REPO, "docs", "field_candidates")
    os.makedirs(outdir, exist_ok=True)

    base = sorted(glob.glob(os.path.join(REPO, "data", "field_tests", "*", "*.wav")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    for cname in CHUNKS:
        path = [p for p in base if os.path.basename(p) == cname][0]
        x, _ = sf.read(path, dtype="float64")           # (n, 64)
        x = x.T                                         # (64, n)
        for d in DIRS:
            outs = {
                "DAS": das.process(x, far_pos(d)),
                "GSC+gn": gsc.process(gain_normalize(x), far_pos(d)),
                "LCMV": lcmv.process(x, far_pos(d)),
                "SubspaceNull": subspace_null(x, d, k=2),
            }
            for meth, y in outs.items():
                y16 = librosa.resample(np.asarray(y).flatten(), orig_sr=FS8,
                                       target_sr=16000)
                if np.max(np.abs(y16)) > 0:
                    y16 = y16 / (np.max(np.abs(y16)) + 1e-9) * 0.9
                res = ast({"raw": y16, "sampling_rate": 16000}, top_k=3)
                top = ", ".join(f"{r['label'][:26]}={r['score']:.2f}" for r in res)
                ton = tonality8k(y)
                sf.write(os.path.join(outdir, f"{cname[:-4]}_az{int(d)}_{meth}.wav"),
                         np.asarray(y) / (np.max(np.abs(y)) + 1e-12) * 0.9, FS8,
                         subtype="PCM_16")
                rows.append(dict(Chunk=cname, Az=d, Method=meth,
                                 Tonality=round(ton, 1), TopClasses=top))
                print(f"[{cname} az={d:4.0f} {meth:12s}] tonal={ton:5.1f} dB  {top}",
                      flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(REPO, "docs", "field_candidates.csv"),
                              index=False)
    print("\nSaved docs/field_candidates.csv + WAVs in docs/field_candidates/")


if __name__ == "__main__":
    main()

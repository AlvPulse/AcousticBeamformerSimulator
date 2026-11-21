"""
NORMALIZATION-RUIN benchmark: how per-chunk data-driven normalization
corrupts the exact feature the company detector consumes.

Company detection model (per user): per-beam POWER HISTORY + anomaly
detection with clustering -- the statistic is the CHANGE of a beam's
level relative to its own history, NOT absolute power. Therefore the
sacred quantity is CROSS-CHUNK LEVEL STABILITY of each beam.

Company LCMV (per user): channels peak-normalized (max -> 1) per chunk
before LCMV; the normalizing constant is set by the loudest content, so
it changes with interference level -- destructive to amplitude values
and to the history statistic.

Test: 60 chunks; target = constant-level tonal at az 40 for the whole
run; interferer = broadband +30 dB at az -60, PRESENT ONLY in chunks
20-39. Normalization policies applied per chunk:
  fixed : no normalization (reference)
  rms   : per-channel RMS normalization (my earlier proposal)
  max   : per-channel peak normalization max->1 (company scheme)
Extractors: DAS and LCMV steered at the target beam (their node).

Metrics:
  - target-beam level series (dB, referenced to chunk 0)
  - level scatter during interferer-ABSENT chunks (should be ~0 dB)
  - level shift at interferer onset: total vs the pure-leakage baseline
    ('fixed' policy) -> the EXTRA shift caused by normalization
  - false-anomaly rate on the TARGET beam: z-score vs its own clean
    history (median/MAD over chunks 0-19), z > 3
  - delivered-audio amplitude fidelity: RMS deviation of the target
    component across same-level chunks (dB)
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from experiments.benchmark_extreme_inr import (
    POS, FS, C, far_pos, frac_delay, steering_far,
)
from experiments.benchmark_phase_robustness import si_sdr

M = POS.shape[0]
N_CHUNKS = 60
CHUNK_S = 2.0
AZ_T, AZ_I = 40.0, -60.0
I_ON = (20, 39)          # interferer present in chunks 20..39
rng = np.random.default_rng(5)

das = StandardDAS(POS, FS)
lcmv = LCMVSMI(POS, FS)


def prop_far(sig, az):
    u = far_pos(az) / 2000.0
    return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])


def make_chunk(k):
    t = np.arange(int(FS * CHUNK_S)) / FS
    tgt = sum(np.sin(2 * np.pi * f * t) for f in (350, 700, 1050)) / 3 * 0.0316
    x = prop_far(tgt, AZ_T)
    if I_ON[0] <= k <= I_ON[1]:
        intf = spsig.filtfilt(*spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)],
                                            btype="band"),
                              rng.standard_normal(len(t)))
        intf = intf * (10 ** (30 / 20) * 0.0316) / np.sqrt(np.mean(intf ** 2))
        x = x + prop_far(intf, AZ_I)
    return x + 0.00316 * rng.standard_normal((M, len(t))), tgt


def norm(x, policy):
    if policy == "fixed":
        return x
    if policy == "rms":
        g = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))
        return x / (g + 1e-12)
    if policy == "max":
        g = np.max(np.abs(x), axis=1, keepdims=True)
        return x / (g + 1e-12)
    raise ValueError(policy)


def main():
    rows = []
    chunks = [make_chunk(k) for k in range(N_CHUNKS)]
    for policy in ("fixed", "rms", "max"):
        for extractor in ("DAS", "LCMV"):
            levels, fid = [], []
            for k, (x, tgt) in enumerate(chunks):
                xn = norm(x, policy)
                y = (das if extractor == "DAS" else lcmv).process(xn, far_pos(AZ_T))
                levels.append(10 * np.log10(np.mean(y ** 2) + 1e-20))
                fid.append(10 * np.log10(np.mean(y ** 2) + 1e-20))
            levels = np.array(levels) - levels[0]
            base = 1.4826 * np.median(np.abs(levels[:I_ON[0]]
                                             - np.median(levels[:I_ON[0]]))) + 1e-6
            med = np.median(levels[:I_ON[0]])
            z = np.abs(levels - med) / base
            false_anom = int(np.sum(z[I_ON[0]:] > 3.0))
            quiet_scatter = float(np.std(levels[:I_ON[0]]))
            onset_shift = float(levels[I_ON[0]])       # total shift at onset
            rows.append(dict(Policy=policy, Extractor=extractor,
                             QuietScatter_dB=quiet_scatter,
                             OnsetShift_dB=onset_shift,
                             FalseAnomalies=false_anom,
                             MaxShift_dB=float(np.max(np.abs(levels)))))
            print(f"[{policy:5s}/{extractor:4s}] quiet scatter={quiet_scatter:5.2f} dB  "
                  f"onset shift={onset_shift:7.2f} dB  false anomalies on TARGET beam="
                  f"{false_anom}/{N_CHUNKS - I_ON[0]}  max|shift|={np.max(np.abs(levels)):7.2f} dB",
                  flush=True)

    # delivered-audio amplitude fidelity: same true target level in every
    # chunk -> deviation of the target-component level in the output
    print("\namplitude fidelity of the delivered target audio (should be 0 dB "
          "deviation; chunk-0 referenced):")
    for policy in ("fixed", "rms", "max"):
        dev = []
        for k in (5, 15, 25, 35, 45, 55):   # before/during/after interferer
            x, tgt = chunks[k]
            xn = norm(x, policy)
            y = das.process(xn, far_pos(AZ_T))
            s = si_sdr(y, tgt)  # relative level proxy via best-fit scale a
            from experiments.benchmark_phase_robustness import align
            ya, ra = align(y, tgt)
            a = np.dot(ya, ra) / (np.dot(ya, ya) + 1e-12)
            dev.append(20 * np.log10(abs(a)))
        dev = np.array(dev) - dev[0]
        print(f"  {policy:5s}: fitted target gain per chunk (dB rel chunk5): "
              f"{np.round(dev, 1)}")
        rows.append(dict(Policy=policy, Extractor="target-gain-dev",
                         QuietScatter_dB=float(np.std(dev)),
                         OnsetShift_dB=float(dev[2]),  # chunk 25 = interferer on
                         FalseAnomalies=np.nan, MaxShift_dB=float(np.max(np.abs(dev)))))

    pd.DataFrame(rows).to_csv(os.path.join(REPO, "docs", "norm_ruin_benchmark.csv"),
                              index=False)


if __name__ == "__main__":
    main()

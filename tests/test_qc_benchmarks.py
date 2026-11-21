"""
QC of the benchmark machinery itself (user-requested): verify the claims
the benchmarks rest on, with independent computations.

1. Capon-map argmax == explicit LCMV grid-sweep argmax (the matrix speedup).
2. SI-SDR: scale-invariant, matches a manual formula.
3. Tonality selector: correct direction on known inputs.
4. Gain round-trip: RMS estimator recovers known gains (far-field scene).
5. Per-chunk normalization stable in quiet scenes (the damage is
   interference-coupled, not stationary).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.benchmark_all_pipelines import (
    gen_tonal, gen_bb, prop_far, capon_map,
)
from experiments.benchmark_phase_robustness import si_sdr
from simulator.beamformers.lcmv_smi import LCMVSMI
from experiments.benchmark_extreme_inr import POS, FS, far_pos, C, frac_delay


def _scene():
    t = np.arange(int(FS * 1.0)) / FS
    rng = np.random.default_rng(0)
    x = prop_far(gen_tonal(t) * 0.1, 40.0) \
        + prop_far(gen_bb(t) * 10 ** (10 / 20) * 0.1 / np.sqrt(np.mean(gen_bb(t) ** 2)), -60.0) \
        + 0.0316 * rng.standard_normal((POS.shape[0], len(t)))
    return x


def test_capon_map_matches_lcmv_sweep():
    mix = _scene()
    grid, pows = capon_map(mix, kind="lcmv", grid=np.arange(-180, 180, 30.0))
    az_capon = float(grid[np.argmax(pows)])
    lcmv = LCMVSMI(POS, FS)
    sweep = {az: float(np.mean(lcmv.process(mix, far_pos(az)) ** 2))
             for az in grid}
    az_sweep = float(max(sweep, key=sweep.get))
    assert abs(az_capon - az_sweep) <= 30.0, (
        f"Capon argmax {az_capon} != LCMV sweep argmax {az_sweep}")


def test_si_sdr_sanity():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000)
    # scale invariance: scaling the estimate must not change SI-SDR
    a = si_sdr(2.0 * x, x)
    b = si_sdr(0.001 * x, x)
    assert a > 60 and b > 60, (a, b)
    # manual cross-check with known noise (same convention: scale the
    # ESTIMATE to the reference, a = <y,x>/<y,y>)
    n = 0.1 * rng.standard_normal(8000)
    y = x + n
    got = si_sdr(y, x)
    alpha = np.dot(y, x) / np.dot(y, y)
    want = 10 * np.log10(np.dot(x, x) / np.dot(alpha * y - x, alpha * y - x))
    assert abs(got - want) < 0.5, (got, want)


def test_tonality_direction():
    from experiments.benchmark_all_pipelines import tonality8k
    t = np.arange(int(FS * 1.0)) / FS
    rng = np.random.default_rng(2)
    tonal = tonality8k(sum(np.sin(2 * np.pi * f * t) for f in (350, 700, 1050)) / 3)
    bb = tonality8k(rng.standard_normal(len(t)))
    assert tonal - bb > 20, (tonal, bb)


def test_gain_roundtrip():
    rng = np.random.default_rng(3)
    t = np.arange(int(FS * 1.0)) / FS
    x = prop_far(gen_tonal(t) * 0.1, 40.0) + 0.0316 * rng.standard_normal((POS.shape[0], len(t)))
    g_true = 10 ** (rng.uniform(-6, 6, POS.shape[0]) / 20)
    xg = x * g_true[:, None]
    g_est = np.sqrt(np.mean(xg ** 2, axis=1))
    err_db = 20 * np.log10(g_est / (g_est.mean() * g_true) * g_true.mean())
    # estimator recovers RELATIVE gains within 0.5 dB
    assert np.max(np.abs(err_db)) < 0.5, err_db


def test_perchunk_norm_stability_quiet():
    from experiments.benchmark_norm_ruin import make_chunk, norm
    gs = [np.sqrt(np.mean(norm(make_chunk(k)[0], "rms") ** 2, axis=1))
          for k in (0, 5, 10)]
    spread_db = 20 * np.log10(np.max(gs) / np.min(gs))
    assert spread_db < 0.5, spread_db


if __name__ == "__main__":
    for fn in (test_capon_map_matches_lcmv_sweep, test_si_sdr_sanity,
               test_tonality_direction, test_gain_roundtrip,
               test_perchunk_norm_stability_quiet):
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL QC PASS")

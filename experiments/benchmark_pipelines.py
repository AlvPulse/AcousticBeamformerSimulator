"""
Pipeline-SYSTEM benchmark (extends benchmark_unified):

Systems compared end-to-end on the same scenes (3 gain conditions x N seeds):
  A      company baseline, frozen: grid LCMV energy-scan -> peak -> audio@peak
  A'     "just swap the algorithm": same grid scan with PhaseOnlyLCMV
  A''    cheapest variant: grid DAS scan
  B      DOA-first: NormMUSIC K=3 -> LCMV extraction at candidates -> harmonic-ID
  B'     DOA-first with imbalance-safe extractor (PhaseOnlyLCMV)
  C      blind: pra.bss.auxiva (OverIVA) multi-output -> manifold DOA -> harmonic-ID
  C'     all repo gridless/blind extractors, single-output + manifold DOA
         (GEV, SVE_LCMV, IterativeLCMV, AuxIVA, FastIVA, ILRMA)

Method coverage matrix: EVERY simulator/ beamformer extracts at the oracle
direction (SI-SDR + purity decomposition + RTF), 3 gain conditions.

Purity metric: least-squares decomposition of the output onto the aligned
clean references -> target-retention / interference-residual / delta-SIR vs mic 0.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig
import pyroomacoustics as pra

from experiments.benchmark_unified import (
    FS, DUR, M, POS, TRUE_DOAS, TARGET_AZ, src_pos, make_scene,
    harmonic_score, si_sdr_aligned, manifold_doa,
)
from simulator.beamformers.das import StandardDAS
from simulator.beamformers.tapered_das import TaperedDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.robust_socp import RobustSOCP
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.ebb import EBB
from simulator.beamformers.multirank_osp import MultiRankOSP
from simulator.beamformers.cgmm_mvdr import cGMM_MVDR
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from simulator.beamformers.sve_lcmv import SVE_LCMV
from simulator.beamformers.gev import GEV
from simulator.beamformers.ilrma import ILRMA
from simulator.beamformers.aux_iva import AuxIVA
from simulator.beamformers.fast_iva import FastIVA
from simulator.beamformers.iterative_lcmv import IterativeLCMV

AZ_GRID = np.arange(-180, 180, 5.0)
FREQS = np.fft.rfftfreq(512, 1 / FS)
N_SEEDS = 3
CONDITIONS = [(0.0, "clean"), (6.0, "gain_6dB"), (12.0, "gain_12dB")]

ALL_GRID_BF = [
    ("TaperedDAS", TaperedDAS), ("LCMVSMI", LCMVSMI), ("RobustLCMV", RobustLCMV),
    ("RobustSOCP", RobustSOCP), ("PhaseOnlyLCMV", PhaseOnlyLCMV), ("EBB", EBB),
    ("MultiRankOSP", MultiRankOSP), ("cGMM_MVDR", cGMM_MVDR), ("GSC_MCWF", GSC_MCWF),
    ("SVE_LCMV", SVE_LCMV),
]
ALL_BLIND = [
    ("GEV", GEV), ("AuxIVA(repo)", AuxIVA), ("FastIVA(repo)", FastIVA),
    ("ILRMA(repo)", ILRMA), ("IterativeLCMV", IterativeLCMV),
]


def _aligned(ref, y):
    """Roll ref to best lag vs y (FFT correlation), trim to common length."""
    n = min(len(ref), len(y))
    c = spsig.correlate(y[:n] - y[:n].mean(), ref[:n] - ref[:n].mean(),
                        mode="full", method="fft")
    lag = int(np.argmax(np.abs(c)) - (n - 1))
    if lag > 0:
        r = ref[lag:lag + len(y)]
    else:
        r = ref[:len(y) + lag] if lag < 0 else ref[:len(y)]
        if lag < 0:
            y = y[-lag:]
    m = min(len(r), len(y))
    return r[:m], y[:m]


def purity(y, refs):
    """LS decomposition onto aligned clean refs.
    Returns (SIR_out_dB, target_retention_dB, interference_resid_dB)."""
    y = np.asarray(y).flatten()
    cols = []
    for r in refs:
        ra, ya = _aligned(np.asarray(r).flatten(), y)
        cols.append((ra, ya))
    n = min(len(c[1]) for c in cols)
    A = np.stack([c[0][:n] for c in cols], axis=1)
    yv = cols[0][1][:n]
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    p = [float(np.sum((coef[i] * A[:, i]) ** 2)) for i in range(A.shape[1])]
    p_t, p_i = p[0], sum(p[1:])
    return (10 * np.log10(p_t / (p_i + 1e-20)),
            10 * np.log10(p_t + 1e-20),
            10 * np.log10(p_i + 1e-20))


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def grid_scan(bf, mix):
    t0 = time.perf_counter()
    pows, aud = [], {}
    for az in AZ_GRID:
        y = bf.process(mix, src_pos(az))
        pows.append(float(np.mean(y ** 2)))
        aud[az] = y
    return np.array(pows), aud, time.perf_counter() - t0


def music_candidates(mix, k=3):
    Z = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
    d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=512,
                          azimuth=np.deg2rad(np.arange(-180, 180, 2.0)))
    d.locate_sources(Z, num_src=k, freq_range=[300.0, 2000.0])
    return np.rad2deg(np.atleast_1d(d.azimuth_recon)).tolist()


def select_by_feature(cands, audios):
    scored = [(harmonic_score(audios[c]), c) for c in cands]
    best = max(scored)[1]
    return best, dict((c, s) for s, c in scored)


def run_pipeline_systems(mix, refs, cond, seed, rows):
    rt, ri1, ri2 = refs
    sir_in, _, _ = purity(mix[0], refs)

    # ---- A family: grid energy-scan (frozen architecture) ----
    for sysname, cls in [("A_gridLCMV", LCMVSMI), ("A'_gridPhaseOnly", PhaseOnlyLCMV),
                         ("A''_gridDAS", StandardDAS)]:
        bf = cls(POS, FS)
        pows, aud, dt = grid_scan(bf, mix)
        peak = float(AZ_GRID[np.argmax(pows)])
        y = aud[peak]
        sisdr = si_sdr_aligned(y, rt)
        sir_o, _, _ = purity(y, refs)
        rows.append(dict(Condition=cond, Seed=seed, System=sysname,
                         SelDOA=peak, DOA_err=ang(peak, TARGET_AZ),
                         SI_SDR=sisdr, dSIR=sir_o - sir_in, SIR_out=sir_o,
                         RTF=dt / DUR))

    # ---- B family: DOA-first + candidate extraction + feature-ID ----
    cands = music_candidates(mix)
    for sysname, cls in [("B_music+LCMV", LCMVSMI), ("B'_music+PhaseOnly", PhaseOnlyLCMV)]:
        bf = cls(POS, FS)
        t0 = time.perf_counter()
        auds = {c: bf.process(mix, src_pos(c)) for c in cands}
        sel, _ = select_by_feature(list(cands), auds)
        dt = time.perf_counter() - t0
        y = auds[sel]
        sisdr = si_sdr_aligned(y, rt)
        sir_o, _, _ = purity(y, refs)
        rows.append(dict(Condition=cond, Seed=seed, System=sysname,
                         SelDOA=sel, DOA_err=ang(sel, TARGET_AZ),
                         SI_SDR=sisdr, dSIR=sir_o - sir_in, SIR_out=sir_o,
                         RTF=(dt + 0.09) / DUR))  # +NormMUSIC cost from unified bench

    # ---- C: OverIVA multi-output ----
    t0 = time.perf_counter()
    Z = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
    Y = pra.bss.auxiva(np.transpose(Z, (2, 1, 0)), n_src=len(TRUE_DOAS), n_iter=50,
                       proj_back=True)
    ys = []
    for k in range(Y.shape[2]):
        _, y_k = spsig.istft(Y[:, :, k].T, fs=FS, nperseg=512, noverlap=256, window="hann")
        ys.append(y_k)
    dt = time.perf_counter() - t0
    scored = [(harmonic_score(y), i) for i, y in enumerate(ys)]
    # select by SI-SDR is oracle; report harmonic-ID (deployable) AND best-source (oracle)
    k_h = max(scored)[1]
    k_o = int(np.argmax([si_sdr_aligned(y, rt) for y in ys]))
    for tag, k in [("C_OverIVA(harmID)", k_h), ("C_OverIVA(oracle-src)", k_o)]:
        doa = manifold_doa(Z, ys[k], FREQS, np.arange(-180, 180, 2.0))
        sisdr = si_sdr_aligned(ys[k], rt)
        sir_o, _, _ = purity(ys[k], refs)
        rows.append(dict(Condition=cond, Seed=seed, System=tag,
                         SelDOA=doa, DOA_err=ang(doa, TARGET_AZ),
                         SI_SDR=sisdr, dSIR=sir_o - sir_in, SIR_out=sir_o,
                         RTF=dt / DUR))

    # ---- C': every repo blind extractor, single output + manifold DOA ----
    for name, cls in ALL_BLIND:
        try:
            bf = cls(POS, FS)
            t0 = time.perf_counter()
            y = bf.process(mix, src_pos(TARGET_AZ))
            dt = time.perf_counter() - t0
            doa = manifold_doa(Z, y, FREQS, np.arange(-180, 180, 2.0))
            sisdr = si_sdr_aligned(y, rt)
            sir_o, _, _ = purity(y, refs)
            rows.append(dict(Condition=cond, Seed=seed, System=f"C'_{name}",
                             SelDOA=doa, DOA_err=ang(doa, TARGET_AZ),
                             SI_SDR=sisdr, dSIR=sir_o - sir_in, SIR_out=sir_o,
                             RTF=dt / DUR))
        except Exception as e:
            rows.append(dict(Condition=cond, Seed=seed, System=f"C'_{name}",
                             SelDOA=np.nan, DOA_err=np.nan, SI_SDR=np.nan,
                             dSIR=np.nan, SIR_out=np.nan, RTF=np.nan))
            print(f"    C' {name} failed: {type(e).__name__}: {e}")


def run_method_matrix(mix, refs, cond, rows):
    rt = refs[0]
    sir_in, _, _ = purity(mix[0], refs)
    for name, cls in ALL_GRID_BF:
        try:
            bf = cls(POS, FS)
            t0 = time.perf_counter()
            y = bf.process(mix, src_pos(TARGET_AZ))
            dt = time.perf_counter() - t0
            sisdr = si_sdr_aligned(y, rt)
            sir_o, _, _ = purity(y, refs)
            rows.append(dict(Condition=cond, Method=name, SI_SDR=sisdr,
                             dSIR=sir_o - sir_in, SIR_out=sir_o, RTF=dt / DUR))
        except Exception as e:
            print(f"    {name} failed: {type(e).__name__}: {e}")
            rows.append(dict(Condition=cond, Method=name, SI_SDR=np.nan,
                             dSIR=np.nan, SIR_out=np.nan, RTF=np.nan))


def main():
    pipe_rows, meth_rows = [], []
    # pipeline systems: N seeds x 3 conditions
    for seed in range(N_SEEDS):
        for spread, cond in CONDITIONS:
            import experiments.benchmark_unified as bu
            old = bu.rng
            bu.rng = np.random.default_rng(100 + seed)
            mix, rt, ri1, ri2 = make_scene(spread, return_refs=True)
            bu.rng = old
            print(f"[pipelines] seed={seed} cond={cond}")
            run_pipeline_systems(mix, (rt, ri1, ri2), cond, seed, pipe_rows)
    # method matrix: 1 draw x 3 conditions (coverage, not statistics)
    for spread, cond in CONDITIONS:
        print(f"[method matrix] cond={cond}")
        mix, rt, ri1, ri2 = make_scene(spread, return_refs=True)
        run_method_matrix(mix, (rt, ri1, ri2), cond, meth_rows)

    pdf = pd.DataFrame(pipe_rows)
    mdf = pd.DataFrame(meth_rows)
    pdf.to_csv(os.path.join(REPO, "docs", "pipeline_benchmark.csv"), index=False)
    mdf.to_csv(os.path.join(REPO, "docs", "method_matrix.csv"), index=False)

    print("\n" + "=" * 100)
    print("PIPELINE SYSTEMS (mean over seeds)  [DOA err deg | SI-SDR dB | dSIR dB | RTF]")
    print("=" * 100)
    summ = pdf.groupby(["Condition", "System"]).mean(numeric_only=True)
    summ = summ[["DOA_err", "SI_SDR", "dSIR", "RTF"]]
    print(summ.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\n" + "=" * 100)
    print("METHOD MATRIX @ oracle direction (SI-SDR / dSIR / RTF)")
    print("=" * 100)
    ms = mdf.groupby(["Condition", "Method"]).mean(numeric_only=True)[["SI_SDR", "dSIR", "RTF"]]
    print(ms.to_string(float_format=lambda v: f"{v:8.2f}"))


if __name__ == "__main__":
    main()

"""
Phase 1 Monte Carlo benchmark - focus: UNCALIBRATED ARRAYS.

Improvements under test:
  1. Per-channel RMS gain normalization before beamforming ("+gn") -- the
     cheap auto-calibration answer to mic imbalance.
  2. B extractor family: LCMV vs GSC-MCWF, with and without gain-norm.
  3. A+ = company grid architecture + gain-norm (keep pipeline, add autocal).
  4. Noise-floor-normalized harmonic-contrast selector (replaces the
     broken harmonic_score that preferred noisy outputs).

Scenes: {anechoic (10 draws), field: wind+RPM-drift (20 draws)} x gains {0,6,12} dB.
Report: selection accuracy (|DOA err| < 10 deg), SI-SDR, dSIR with 95% CI.
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

import experiments.benchmark_unified as bu
from experiments.benchmark_unified import (
    FS, DUR, M, POS, TRUE_DOAS, TARGET_AZ, src_pos, si_sdr_aligned, manifold_doa,
)
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.gsc_mcwf import GSC_MCWF

AZ_GRID = np.arange(-180, 180, 5.0)
FREQS = np.fft.rfftfreq(512, 1 / FS)
FREQS[0] = 1e-6
GAINS = [(0.0, "clean"), (6.0, "gain_6dB"), (12.0, "gain_12dB")]
SCENES = [("anechoic", 10, dict()), ("field_wind_drift", 20, dict(wind=True, drift=True))]


def harmonic_contrast(audio):
    """Tonality: peak-to-median ratio of the time-averaged power spectrum,
    within the 200-2500 Hz band (dB). Tonal targets (drone harmonics) score
    high; broadband interferers score low. Noise-floor-normalized by the
    median, so it stays valid on cleanly separated outputs (unlike
    harmonic_score, which scored the clean target at -74 dB)."""
    f, t, S = spsig.spectrogram(np.asarray(audio).flatten(), FS, nperseg=2048,
                                noverlap=1024, window="hann")
    band = (f >= 200) & (f <= 2500)
    P = np.mean(S, axis=1)[band]
    return float(10 * np.log10(np.max(P) / (np.median(P) + 1e-15)))


def gain_normalize(mix):
    """Per-channel RMS normalization: cheap auto-calibration for gain imbalance."""
    r = np.sqrt(np.mean(mix ** 2, axis=1, keepdims=True))
    return mix / (r + 1e-12)


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def purity_dsir(y, refs, sir_in):
    from experiments.benchmark_pipelines import purity
    sir_o, _, _ = purity(y, refs)
    return sir_o - sir_in


def music_candidates(mix, k=3):
    Z = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
    d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=512,
                          azimuth=np.deg2rad(np.arange(-180, 180, 2.0)))
    d.locate_sources(Z, num_src=k, freq_range=[300.0, 2000.0])
    return np.rad2deg(np.atleast_1d(d.azimuth_recon)).tolist()


def sweep(bf, mix):
    t0 = time.perf_counter()
    pows, aud = [], {}
    for az in AZ_GRID:
        y = bf.process(mix, src_pos(az))
        pows.append(float(np.mean(y ** 2)))
        aud[az] = y
    return np.array(pows), aud, time.perf_counter() - t0


def run_draw(mix, refs, scene, gain, draw, rows, with_a_family=False):
    rt = refs[0]
    sir_in = None
    from experiments.benchmark_pipelines import purity
    sir_in, _, _ = purity(mix[0], refs)

    def rec(system, sel, y, rtf):
        sisdr = si_sdr_aligned(y, rt)
        dsir = purity_dsir(y, refs, sir_in)
        rows.append(dict(Scene=scene, Gain=gain, Draw=draw, System=system,
                         SelDOA=sel, Err=ang(sel, TARGET_AZ),
                         Correct=int(ang(sel, TARGET_AZ) < 10.0),
                         SI_SDR=sisdr, dSIR=dsir, RTF=rtf))

    mix_gn = gain_normalize(mix)

    # --- A family: grid energy-scan architecture (field scene only) ---
    if with_a_family:
        for sysname, bf, x in [("A_gridLCMV", LCMVSMI(POS, FS), mix),
                               ("A'_gridPhaseOnly", PhaseOnlyLCMV(POS, FS), mix),
                               ("A+_gridLCMV+gn", LCMVSMI(POS, FS), mix_gn)]:
            pows, aud, dt = sweep(bf, x)
            peak = float(AZ_GRID[np.argmax(pows)])
            rec(sysname, peak, aud[peak], dt / DUR)

    # --- B family: DOA-first + candidate extraction + contrast-ID ---
    cands = music_candidates(mix)
    for sysname, make, x in [
            ("B_music+LCMV", lambda: LCMVSMI(POS, FS), mix),
            ("B2_music+GSC", lambda: GSC_MCWF(POS, FS), mix),
            ("B3_music+GSC+gn", lambda: GSC_MCWF(POS, FS), mix_gn)]:
        bf = make()
        t0 = time.perf_counter()
        auds = {c: bf.process(x, src_pos(c)) for c in cands}
        sel = max(cands, key=lambda c: harmonic_contrast(auds[c]))
        rec(sysname, sel, auds[sel], (time.perf_counter() - t0 + 0.09) / DUR)

    # --- C: OverIVA multi-output + contrast-ID ---
    t0 = time.perf_counter()
    Z = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
    Y = pra.bss.auxiva(np.transpose(Z, (2, 1, 0)), n_src=len(TRUE_DOAS), n_iter=50,
                       proj_back=True)
    ys = []
    for k in range(Y.shape[2]):
        _, y_k = spsig.istft(Y[:, :, k].T, fs=FS, nperseg=512, noverlap=256, window="hann")
        ys.append(y_k)
    dt = time.perf_counter() - t0
    k_sel = int(np.argmax([harmonic_contrast(y) for y in ys]))
    doa = manifold_doa(Z, ys[k_sel], FREQS, np.arange(-180, 180, 2.0))
    rec("C_OverIVA(contrastID)", doa, ys[k_sel], dt / DUR)


def main():
    # one-time selector sanity check on the clean references
    mix, rt, ri1, ri2 = bu.make_scene(0.0, return_refs=True)
    print("harmonic_contrast sanity: target=%.1f dB  noise1=%.1f dB  noise2=%.1f dB"
          % (harmonic_contrast(rt), harmonic_contrast(ri1), harmonic_contrast(ri2)))

    rows = []
    for scene, n_draws, kw in SCENES:
        for spread, gain in GAINS:
            for draw in range(n_draws):
                bu.rng = np.random.default_rng(500 + draw)
                mix, rt, ri1, ri2 = bu.make_scene(spread, return_refs=True, **kw)
                print(f"[mc] scene={scene} gain={gain} draw={draw}", flush=True)
                run_draw(mix, (rt, ri1, ri2), scene, gain, draw, rows,
                         with_a_family=(scene.startswith("field")))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "mc_benchmark.csv"), index=False)

    g = df.groupby(["Scene", "Gain", "System"])
    summ = g.agg(SelectAcc=("Correct", "mean"), SI_SDR=("SI_SDR", "mean"),
                 SI_SDR_ci=("SI_SDR", lambda s: 1.96 * s.std() / np.sqrt(len(s))),
                 dSIR=("dSIR", "mean"), RTF=("RTF", "median")).reset_index()
    print("\n" + "=" * 100)
    print("MONTE CARLO RESULTS  [SelectAcc | SI-SDR +/-95%CI dB | dSIR dB | median RTF]")
    print("=" * 100)
    for (scene, gain), sub in summ.groupby(["Scene", "Gain"]):
        print(f"\n--- {scene} @ {gain} ---")
        for _, r in sub.iterrows():
            print(f"  {r.System:24s} acc={r.SelectAcc:5.0%}  "
                  f"SI-SDR={r.SI_SDR:6.2f}+/-{r.SI_SDR_ci:5.2f}  "
                  f"dSIR={r.dSIR:7.2f}  rtf={r.RTF:.2f}")


if __name__ == "__main__":
    main()

"""
Gap closers (validated before Phase 2):

GAP 1 — Multi-pitch estimator for the aligned-hum case.
  Greedy f0 tracking fails when a loud tonal masks a weak one in the same
  beam (siren recovery 40-60%). Replacement: enumerate spectral peaks as
  f0 hypotheses, score each by HARMONIC CONSISTENCY (sum of line powers at
  f0,2f0,3f0 normalized by local background), keep the best two
  hypotheses whose harmonic combs do not overlap. The weak-but-clean
  siren (3 clean harmonics) outscores noise peaks even when the hum is
  +45 dB above it.

GAP 2 — Multi-chunk narrowband integration for static-continuous
  interference. Single-chunk MUSIC cannot see a target 70 dB under a
  continuous machine. Averaging the narrowband spatial covariance across
  N chunks reduces estimation variance ~N -> the spurious-peak floor
  drops -> the weak coherent source emerges. Target-persistence prior
  (allowed); no interferer prior.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig

from simulator.beamformers.das import StandardDAS
from experiments.benchmark_extreme_inr import (
    POS, FS, DUR, C, NFFT, far_pos, frac_delay, steering_far,
)
from experiments.benchmark_blind_mc import make_scene as make_scene3, ang
from experiments.benchmark_blind_extraction import comb, band_power_db, SIREN_LINES, URBAN_LINES

M = POS.shape[0]
SIREN_F0 = 350.0


# --------------------------------------------------------------------------
def multipitch(y, fmin=150.0, fmax=1100.0, n_keep=2):
    """Enumerate f0 hypotheses from spectral peaks; score by harmonic
    consistency (line power at f0 x h vs LOCAL background). Returns up to
    n_keep (f0, score_db) with non-overlapping combs."""
    f, P = spsig.periodogram(np.asarray(y).flatten(), FS, nfft=16384,
                             window="blackman")
    band = (f >= fmin) & (f <= fmax)
    fb, Pb = f[band], P[band]
    pk, _ = spsig.find_peaks(Pb)
    if len(pk) == 0:
        return []
    top = pk[np.argsort(Pb[pk])[::-1][:30]]

    def line_power(fc, bw=8.0):
        sel = np.abs(f - fc) < bw
        return float(np.sum(P[sel])) if np.any(sel) else 0.0

    def local_bg(fc, bw=150.0):
        sel = (np.abs(f - fc) < bw) & (np.abs(f - fc) > 30.0)
        return float(np.mean(P[sel])) + 1e-20

    # hypothesis set: peaks AND their subharmonics (a weak fundamental may
    # never peak while its 2nd/3rd harmonic does -- the aligned-hum failure)
    cand = set()
    for i in top:
        f0 = fb[i]
        for k in (1, 2, 3):
            if f0 / k >= fmin:
                cand.add(round(f0 / k, 2))
    # frame-wise persistence: a line must sit at the SAME frequency in
    # >=70% of short frames (stationary tonal) -- wandering broadband
    # bumps fail this even when their time-averaged excess is large
    n_fr = 16
    fl, tl, SL = spsig.spectrogram(np.asarray(y).flatten(), FS, nperseg=2048,
                                   noverlap=1024, window="hann")

    def occupancy(fc, bw=12.0):
        sel = np.abs(fl - fc) < bw
        if not np.any(sel):
            return 0.0
        band = SL[sel, :]                       # (lines, frames)
        peak_f = fl[sel][np.argmax(band.mean(axis=1))]
        frame_peak = fl[sel][np.argmax(band, axis=0)]
        return float(np.mean(np.abs(frame_peak - peak_f) < 4.0))

    hyps = []
    for f0 in sorted(cand):
        if f0 < fmin:
            continue
        excess_sum, strong, n_lines = 0.0, 0, 0
        for h in (1, 2, 3):
            fh = f0 * h
            if fh > FS / 2 - 50:
                break
            n_lines += 1
            excess = line_power(fh) / local_bg(fh)
            if excess > 4.0 and occupancy(fh) >= 0.5:   # tonal; envelope may duty-cycle
                strong += 1
                excess_sum += excess
        if strong >= 2 and n_lines >= 3:
            hyps.append((float(f0), 10 * np.log10(excess_sum / n_lines)))
    hyps.sort(key=lambda h: -h[1])

    kept = []
    for f0, s in hyps:
        ok = True
        for f0k, _s in kept:
            for h1 in (1, 2, 3):
                for h2 in (1, 2, 3):
                    if abs(f0 * h1 - f0k * h2) < 15.0:
                        ok = False
        if ok:
            kept.append((f0, s))
        if len(kept) == n_keep:
            break
    return kept


def gap1_aligned_hum(n_draws=10):
    """B45 scene: hum aligned with the siren. Greedy vs multipitch."""
    print("=" * 84)
    print("GAP 1 - aligned hum: multi-pitch vs greedy f0 tracking (siren recovery)")
    print("=" * 84)
    das = StandardDAS(POS, FS)
    greedy_ok, mp_ok, margins = 0, 0, []
    for draw in range(n_draws):
        rng = np.random.default_rng(9000 + draw)
        az_s = float(rng.uniform(-170, 170))
        az_u = az_s  # aligned
        az_b = (az_s + float(rng.choice([-1, 1]) * rng.uniform(60, 150))) % 360
        az_b = az_b - 360 * (az_b > 180)
        seed = 9100 + draw
        mix, *_ = make_scene3(seed, az_s, az_b, az_u, 45.0)
        y = das.process(mix, far_pos(az_s))  # merged beam

        from experiments.benchmark_blind_extraction import track_f0, notch_lines
        f0_g, _ = track_f0(y)                    # greedy: strongest tonal
        greedy_ok += int(f0_g is not None and abs(f0_g - SIREN_F0) < 15)

        hyps = multipitch(y)
        sirenh = [h for h in hyps if abs(h[0] - SIREN_F0) < 15]
        mp_ok += int(len(sirenh) > 0)
        if sirenh:
            yc = comb(y, SIREN_F0)
            m = (band_power_db(yc, freq_list=SIREN_LINES)
                 - max(band_power_db(yc, freq_list=URBAN_LINES),
                       band_power_db(yc, band_list=[(1550, 1750)])))
            margins.append(m)
        print(f"  draw {draw}: greedy f0={f0_g and round(f0_g,1)}  "
              f"multipitch={[round(h[0],1) for h in hyps]}"
              f"{'  siren margin %.1f dB' % margins[-1] if sirenh else ''}",
              flush=True)
    print(f"RESULT: greedy siren-recovery {greedy_ok}/{n_draws}  |  "
          f"multi-pitch {mp_ok}/{n_draws}"
          f"  |  median margin {np.median(margins) if margins else float('nan'):.1f} dB")


# --------------------------------------------------------------------------
def narrowband_music(R, A_grid):
    """Subspace MUSIC for one frequency bin given covariance R (M,M) and
    steering vectors A_grid (n_grid, M)."""
    evals, evecs = np.linalg.eigh(R)
    n_src = 4  # near-field broadband spans 2 eigvecs; +hum +weak target
    En = evecs[:, :-n_src]
    P = np.array([1.0 / (np.abs(a.conj() @ En @ En.conj().T @ a) + 1e-15)
                  for a in A_grid])
    return P


def gap2_static_continuous(n_chunks=60):
    """C50 scene repeated across chunks. Blind recipe: (1) wideband MUSIC
    finds the LOUD directions (machine/bark, urban); (2) average the
    narrowband covariance at the siren line over chunks; (3) project the
    loud steerings OUT; (4) rank-1 Bartlett scan on the residual -- the
    weak coherent source is what remains."""
    import pyroomacoustics as pra
    print("=" * 84)
    print(f"GAP 2 - static-continuous +50 dB: loud-projected {n_chunks}-chunk "
          "narrowband scan @ siren line")
    print("=" * 84)
    az_s = 40.0
    fgrid = np.fft.rfftfreq(NFFT, 1 / FS)
    f0_bin = int(np.argmin(np.abs(fgrid - SIREN_F0)))
    azs = np.arange(-180, 180, 1.0)
    A_list = np.stack([steering_far(a)[f0_bin, :] for a in azs])  # (G,M)

    R_sum = None
    loud_az = None
    for c in range(n_chunks):
        mix, *_ = make_scene3(20000 + c, az_s, -60.0, 120.0, 50.0)
        Z = spsig.stft(mix, fs=FS, nperseg=NFFT, noverlap=NFFT - 256,
                       window="hann")[2]
        if loud_az is None:  # once: blind wideband DOA for loud sources
            d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=NFFT,
                                  azimuth=np.deg2rad(np.arange(-180, 180, 2.0)))
            d.locate_sources(Z, num_src=3, freq_range=[250.0, 1100.0])
            est = np.rad2deg(np.atleast_1d(d.azimuth_recon))
            # loud directions = estimates far from... blindly: all of them
            # except we take those with the strongest peaks: keep all 3,
            # siren is too weak to appear here anyway
            loud_az = [float(e) for e in est]
            print(f"  blind wideband MUSIC loud candidates: {np.round(loud_az,1)}"
                  f"  (truth: bark -60, urban 120, siren 40 weak)")
        X = Z[:, f0_bin, :]
        R = (X @ X.conj().T) / X.shape[1]
        R_sum = R if R_sum is None else R_sum + R
    R_avg = R_sum / n_chunks

    # project out the loud steerings (incl. near-field rank-2 spread: null
    # each loud direction AND a 2nd basis vector from its local covariance)
    P = np.eye(M, dtype=complex)
    for az in loud_az:
        a = steering_far(az)[f0_bin, :]
        P -= np.outer(a, a.conj()) / (a.conj() @ a)
    # also remove the top-2 eigvecs of the projected covariance (rank
    # spread of the near-field loud source)
    Rp = P @ R_avg @ P
    evals, evecs = np.linalg.eigh(Rp)
    for k in range(1, 3):
        v = evecs[:, -k]
        Rp = Rp - np.outer(v, v.conj()) * (v.conj() @ Rp @ v).real
    Pw = np.eye(M, dtype=complex)
    for k in range(1, 3):
        v = evecs[:, -k]
        Pw -= np.outer(v, v.conj())
    R_res = Pw @ R_avg @ Pw

    bartlett = np.array([np.real(a.conj() @ R_res @ a) for a in A_list])
    top5 = azs[np.argsort(bartlett)[::-1][:5]]
    hit = any(ang(t, az_s) < 15 for t in top5)
    print(f"  loud-projected residual scan: top-5 az = {np.round(top5,0)} "
          f"(siren truth {az_s})  -> siren found: {hit}")


if __name__ == "__main__":
    gap1_aligned_hum()
    gap2_static_continuous()

"""
First pass over the real field recordings (data/field_tests): QC + blind DOA.

1. Hardware QC (no calibration exists, so measure it):
   - per-channel RMS level spread (the REAL gain imbalance of this system)
   - dead/quiet/clipped channels, DC offsets
2. Blind per-chunk DOA: NormMUSIC + SRP, 64 mics, 250-1100 Hz, K=3.
3. Cross-chunk persistence: azimuth histogram -> directions that persist
   across chunks (first track-layer output; no ground truth exists, so
   consistency is the evidence).
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
import pyroomacoustics as pra

FS = 8000
NFFT = 512
POS = np.loadtxt(os.path.join(REPO, "array_geo.csv"), delimiter=",").T  # (64,3)
M = POS.shape[0]

CHUNKS = sorted(glob.glob(os.path.join(REPO, "data", "field_tests", "*", "*.wav")),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main():
    assert len(CHUNKS) > 0, "no chunks found in data/field_tests"
    print(f"field pass: {len(CHUNKS)} chunks, {M} mics, fs={FS}")

    # ---------- 1. hardware QC ----------
    rms_db_all, dc_all, clip_all = [], [], []
    for f in CHUNKS:
        x, fs = sf.read(f, dtype="float64")
        assert fs == FS and x.shape[1] == M, (f, fs, x.shape)
        rms_db_all.append(20 * np.log10(np.sqrt(np.mean(x ** 2, axis=0)) + 1e-12))
        dc_all.append(np.mean(x, axis=0))
        clip_all.append(np.max(np.abs(x), axis=0) >= 0.999)

    rms_db = np.array(rms_db_all)                     # (n_chunks, M)
    ch_med = np.median(rms_db, axis=0)                # per-channel level
    spread = np.max(ch_med) - np.min(ch_med)
    med = np.median(ch_med)
    quiet = np.where(ch_med < med - 20)[0]
    loud = np.where(ch_med > med + 20)[0]
    dead = np.where(ch_med < med - 40)[0]
    clipping = np.where(np.sum(clip_all, axis=0) > len(CHUNKS) * 0.5)[0]
    dc_max = np.max(np.abs(np.array(dc_all)))

    print("\n=== HARDWARE QC (measured, no calibration assumed) ===")
    print(f"  per-channel level spread (median over chunks): {spread:.1f} dB "
          f"[min {ch_med.min():.1f}, max {ch_med.max():.1f}]")
    print(f"  channels >20 dB off median: quiet={list(quiet)} loud={list(loud)}")
    print(f"  dead (<-40 dB): {list(dead)}   clipping>50% chunks: {list(clipping)}")
    print(f"  max |DC offset|: {dc_max:.4f}")

    # ---------- 2. blind DOA per chunk ----------
    rows = []
    az_grid = np.deg2rad(np.arange(-180, 180, 2.0))
    for f in CHUNKS:
        name = os.path.basename(f)
        x, _ = sf.read(f, dtype="float64")
        Z = spsig.stft(x.T, fs=FS, nperseg=NFFT, noverlap=NFFT - 256,
                       window="hann")[2]
        res = {"Chunk": name}
        for meth, cls in (("NormMUSIC", pra.doa.NormMUSIC), ("SRP", pra.doa.SRP)):
            kw = {"nfft": NFFT, "mode": "far"} if meth == "SRP" else {"nfft": NFFT}
            try:
                d = cls(POS.T.copy(), FS, azimuth=az_grid, **kw)
                d.locate_sources(Z, num_src=3, freq_range=[250.0, 1100.0])
                est = np.rad2deg(np.atleast_1d(d.azimuth_recon))
                for k in range(3):
                    res[f"{meth}_{k+1}"] = round(float(est[k % len(est)]), 1)
            except Exception as e:
                print(f"  {meth} failed on {name}: {e}")
        rows.append(res)
        m = [res.get(f"NormMUSIC_{k+1}") for k in range(3)]
        s = [res.get(f"SRP_{k+1}") for k in range(3)]
        agree = min((min(ang(a, b) for b in s if b is not None) for a in m
                     if a is not None), default=np.nan)
        print(f"  {name}: MUSIC={m}  SRP={s}  cross-agree={agree:.0f} deg", flush=True)
        rows[-1]["CrossAgree"] = agree

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "field_quick_pass.csv"), index=False)

    # ---------- 3. cross-chunk persistence ----------
    print("\n=== PERSISTENT DIRECTIONS (MUSIC candidates, 15 deg bins, "
          f"{len(CHUNKS)} chunks) ===")
    ests = df[[f"NormMUSIC_{k+1}" for k in range(3)]].to_numpy().ravel()
    ests = ests[~np.isnan(ests)]
    bins = np.arange(-180, 181, 15)
    hist, edges = np.histogram(ests, bins=bins)
    order = np.argsort(hist)[::-1]
    for i in order[:6]:
        if hist[i] == 0:
            break
        ctr = (edges[i] + edges[i + 1]) / 2
        srp_near = 0
        for s in df[[f"SRP_{k+1}" for k in range(3)]].to_numpy().ravel():
            if not np.isnan(s) and ang(s, ctr) < 15:
                srp_near += 1
        print(f"  az ~ {ctr:6.0f} deg: seen in {hist[i]:3d}/{len(ests)} candidates "
              f"({hist[i]/len(ests):4.0%}),  SRP supports {srp_near} times")

    print(f"\nMean cross-estimator agreement: {df['CrossAgree'].mean():.1f} deg")


if __name__ == "__main__":
    main()

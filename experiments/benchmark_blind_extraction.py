"""
BLIND extreme-INR extraction: no oracle steering, no oracle f0, urban noise.

Scene (64-mic real geometry, 8 kHz):
  - siren: harmonics 350/700/1050 Hz, 2 km, az +40  (the weak target, -20 dB
    below sensor noise, with bark +45 dB over it)
  - bark: broadband bursts 300-3000 Hz, 10 m NEAR-FIELD, az -60, +45 dB
  - urban: broadband + 430/860 Hz hum tones, az +120 (scene A) or az +40
    ALIGNED with the siren (scene B: tracker stress), +25 dB

Blind pipeline (nothing knows which source is which):
  1. NormMUSIC (num_src=3) -> candidate azimuths
  2. Per candidate, extract with THREE interchangeable beamformers and pick
     by measured tonality (harmonic_contrast):
       DAS | SubspaceNull(k=2) | LCMV-null-at-other-candidates
  3. Adaptive post-filter: track f0 from the candidate's OWN spectrum
     (peak-picking + harmonic-sum score); apply comb only if tonal
  4. Evaluate per candidate: az error, extractor chosen, tracked f0,
     cross-leak margins vs every other source's characteristic bands
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig
import pyroomacoustics as pra

from simulator.beamformers.das import StandardDAS
from experiments.benchmark_extreme_inr import (
    POS, FS, DUR, C, NFFT, HOP, far_pos, frac_delay, steering_far,
    stft, istft, subspace_null,
)
from experiments.benchmark_mc import harmonic_contrast

M = POS.shape[0]
SIREN_F0 = 350.0
rng = np.random.default_rng(11)

# characteristic bands for leak accounting (evaluation only, not used blind)
SIREN_LINES = [350.0, 700.0, 1050.0]
URBAN_LINES = [430.0, 860.0]
BARK_BANDS = [(1550, 1750), (2100, 2300), (2600, 2800)]  # bark-only zones


def band_power_db(y, freq_list=None, band_list=None, bw=12.0):
    f, P = spsig.periodogram(np.asarray(y).flatten(), FS, nfft=32768,
                             window="blackman")
    mask = np.zeros_like(f, bool)
    if freq_list:
        for fc in freq_list:
            mask |= np.abs(f - fc) < bw
    if band_list:
        for lo, hi in band_list:
            mask |= (f >= lo) & (f <= hi)
    return 10 * np.log10(float(np.sum(P[mask])) + 1e-20)


def make_scene(urban_az):
    t = np.arange(int(FS * DUR)) / FS
    env = 0.5 + 0.5 * np.clip(0.8 * np.sin(2 * np.pi * 0.9 * t + 0.4), -1, 1)
    siren = sum(np.sin(2 * np.pi * f0 * t) for f0 in SIREN_LINES) / 3.0 * env
    siren *= 10 ** (-20 / 20)

    bark_src = np.zeros(len(t))
    b, a = spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)], btype="band")
    for start in (0.3, 0.9, 1.1, 1.7, 2.3, 2.5):
        n = int(0.12 * FS)
        i0 = int(start * FS)
        burst = rng.standard_normal(min(n, len(t) - i0))
        bark_src[i0:i0 + len(burst)] += burst
    bark_src = spsig.filtfilt(b, a, bark_src)

    urban_src = 0.6 * rng.standard_normal(len(t))
    b2, a2 = spsig.butter(4, [250 / (FS / 2), 1500 / (FS / 2)], btype="band")
    urban_src = spsig.filtfilt(b2, a2, urban_src)
    urban_src += 0.8 * (np.sin(2 * np.pi * 430 * t) + 0.6 * np.sin(2 * np.pi * 860 * t))

    def prop_far(sig, az):
        u = far_pos(az) / 2000.0
        return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])

    def prop_near(sig, az, dist=10.0):
        d = np.linalg.norm(POS - far_pos(az, dist), axis=1)
        x = np.stack([frac_delay(sig, d[m] / C) for m in range(M)])
        return x  # external multiplier sets the true per-mic INR (no 1/r)

    x_s = prop_far(siren, 40.0)
    x_b = prop_near(bark_src, -60.0) * 10 ** (45 / 20) * 0.1 / np.sqrt(np.mean(bark_src ** 2))
    x_u = prop_far(urban_src, urban_az) * 10 ** (25 / 20) * 0.1 / np.sqrt(np.mean(urban_src ** 2))
    noise = 0.1 * rng.standard_normal((M, len(t)))
    return x_s + x_b + x_u + noise, siren, bark_src, urban_src


def music_candidates(mix, k=3):
    Z = spsig.stft(mix, fs=FS, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")[2]
    d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=NFFT,
                          azimuth=np.deg2rad(np.arange(-180, 180, 2.0)))
    d.locate_sources(Z, num_src=k, freq_range=[250.0, 1100.0])
    return np.rad2deg(np.atleast_1d(d.azimuth_recon)).tolist()


def lcmv_null_others(mix, az_own, az_others):
    """Classic interchangeable extraction: unity at own azimuth, hard nulls
    at the other candidate azimuths."""
    Z = stft(mix)
    A_own = steering_far(az_own)
    A_oth = [steering_far(a) for a in az_others]
    Y = np.zeros_like(Z[0])
    _, F, T = Z.shape
    n_c = 1 + len(A_oth)
    for f in range(F):
        X = Z[:, f, :]
        R = (X @ X.conj().T) / T
        evals, evecs = np.linalg.eigh(R)
        evals = np.maximum(evals, 1e-3 * float(np.max(evals)))
        Rinv = (evecs * (1.0 / evals)) @ evecs.conj().T
        Amat = np.stack([A_own[f]] + [Ao[f] for Ao in A_oth], axis=1)  # (M, n_c)
        rhs = Rinv @ Amat
        W = rhs @ np.linalg.pinv(Amat.conj().T @ rhs)                   # (M, n_c)
        Y[f] = W[:, 0].conj() @ X
    return istft(Y)


def track_f0(y, fmin=150.0, fmax=1100.0):
    """Blind f0: peak candidates scored by harmonic sum vs band median."""
    f, P = spsig.periodogram(np.asarray(y).flatten(), FS, nfft=8192,
                             window="blackman")
    band = (f >= fmin) & (f <= fmax)
    fb, Pb = f[band], P[band]
    pk, _ = spsig.find_peaks(Pb)
    if len(pk) == 0:
        return None, 0.0
    top = pk[np.argsort(Pb[pk])[::-1][:12]]
    med = float(np.median(Pb)) + 1e-20
    best_f0, best_score = None, 0.0
    for i in top:
        f0 = fb[i]
        if f0 < fmin:
            continue
        score = 0.0
        for h in (1, 2, 3, 4):
            fh = f0 * h
            if fh > FS / 2 - 50:
                break
            sel = np.abs(f - fh) < 8.0
            score += float(np.max(P[sel])) if np.any(sel) else 0.0
        score_db = 10 * np.log10(score / (4 * med))
        if score_db > best_score:
            best_f0, best_score = f0, score_db
    return best_f0, best_score


def comb(y, f0, floor=0.05, width_hz=25.0):
    Z = stft(y[np.newaxis, :])[0]
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    mask = np.full((len(f), Z.shape[1]), floor)
    for h in (1, 2, 3, 4):
        fh = f0 * h
        if fh > f[-1]:
            break
        mask[np.abs(f - fh) < width_hz] = 1.0
    return istft(Z * mask)


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def notch_lines(y, f0, width_hz=40.0):
    Z = stft(y[np.newaxis, :])[0]
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    mask = np.ones((len(f), Z.shape[1]))
    for h in (1, 2, 3, 4):
        fh = f0 * h
        if fh > f[-1]:
            break
        mask[np.abs(f - fh) < width_hz] = 0.0
    return istft(Z * mask)


def full_path(y):
    """Track f0 on y, comb if tonal. Returns (f0, y_final, contrast, comb_used)."""
    f0, score = track_f0(y)
    if f0 is None or score < 6.0:
        return None, y, harmonic_contrast(y), False
    yc = comb(y, f0)
    return f0, yc, harmonic_contrast(yc), True


def match_output(az, f0, truths_az, urban_az):
    """Identify an output by (az, f0): tonal outputs match by f0, others by az."""
    if f0 is not None:
        for src, ref_f0 in (("siren", SIREN_F0), ("urban", 430.0)):
            for h in (1, 2, 3):
                if abs(f0 - ref_f0 * h) < 15:
                    return src
        return f"tonal@{f0:.0f}Hz"
    return min(truths_az, key=lambda k: ang(az, truths_az[k]))


def run_blind(mix, refs, scene_name, rows):
    truths_az = {"siren": 40.0, "bark": -60.0, "urban": URBAN_AZ[scene_name]}
    das = StandardDAS(POS, FS)

    cands = music_candidates(mix)
    print(f"[{scene_name}] blind NormMUSIC candidates: {np.round(cands,1)} "
          f"(truths: siren 40, bark -60, urban {truths_az['urban']})", flush=True)

    for az in cands:
        others = [c for c in cands if ang(c, az) > 15]
        raws = {
            "DAS": das.process(mix, far_pos(az)),
            "SubspaceNull": subspace_null(mix, az, k=2),
            "LCMVnull": lcmv_null_others(mix, az, others),
        }
        # full-path scoring: the extractor whose tracked+combed output is
        # most tonal wins (spatial suppression must EARN its place)
        paths = {n: full_path(y) for n, y in raws.items()}
        chosen = max(paths, key=lambda n: paths[n][2])
        f0_a, y_a, _, used_a = paths[chosen]
        y_raw = raws[chosen]

        outputs = [(f0_a, y_a, chosen, used_a)]
        # second spectral source hiding in the same beam: notch f0_a, re-track
        if f0_a is not None:
            y_res = notch_lines(y_raw, f0_a)
            f0_b, score_b = track_f0(y_res)
            if f0_b is not None and score_b > 3.0 and abs(f0_b - f0_a) > 20:
                outputs.append((f0_b, comb(y_res, f0_b), chosen, True))

        # broadband emission: if most energy survives notching ALL found
        # tonal lines, this beam also carries a broadband source (e.g. bark)
        f0s_found = [o[0] for o in outputs if o[0]]
        y_bb = raws["LCMVnull"]
        for f0_n in f0s_found:
            y_bb = notch_lines(y_bb, f0_n)
        if np.sum(y_bb ** 2) > 0.5 * (np.sum(raws["LCMVnull"] ** 2) + 1e-20):
            outputs.append((None, y_bb, "LCMVnull", False))

        for f0_o, y_o, ext, used in outputs:
            ident = match_output(az, f0_o, truths_az, truths_az["urban"])
            own_char = {"siren": (SIREN_LINES, None), "urban": (URBAN_LINES, None),
                        "bark": (None, BARK_BANDS)}.get(ident)
            if own_char is None:
                continue  # unidentified tonal output: report skipped
            own_p = band_power_db(y_o, freq_list=own_char[0], band_list=own_char[1])
            margins = {}
            for other, (fl, bl) in {
                    "siren": (SIREN_LINES, None), "urban": (URBAN_LINES, None),
                    "bark": (None, BARK_BANDS)}.items():
                if other == ident:
                    continue
                margins[f"vs_{other}"] = own_p - band_power_db(y_o, freq_list=fl, band_list=bl)
            rows.append(dict(Scene=scene_name, Out_az=round(az, 1), Matched=ident,
                             F0=f0_o, Extractor=ext, Comb=used, OwnPower=own_p,
                             **margins))
            print(f"  out az={az:6.1f} -> {ident:6s}  extractor={ext:11s} "
                  f"f0={f0_o if f0_o else float('nan'):6.1f} comb={used}  "
                  + "  ".join(f"{k}={v:+.1f}dB" for k, v in margins.items()), flush=True)


def main():
    global URBAN_AZ
    URBAN_AZ = {"A_separated": 120.0, "B_urban_aligned": 40.0}
    rows = []
    for scene_name in ("A_separated", "B_urban_aligned"):
        mix, siren, bark, urban = make_scene(URBAN_AZ[scene_name])
        run_blind(mix, (siren, bark, urban), scene_name, rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "blind_extraction_benchmark.csv"), index=False)
    print("\nSaved docs/blind_extraction_benchmark.csv")


if __name__ == "__main__":
    main()

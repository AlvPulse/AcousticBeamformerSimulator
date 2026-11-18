"""
Statistical validation of the BLIND extraction pipeline (multi-draw MC).

One draw is not decision-grade: randomize per draw the source azimuths,
envelopes, bark burst timing, and noise; run the full blind pipeline
(NormMUSIC -> per-beam full-path extraction -> adaptive multi-f0 +
broadband emission); score explicit success criteria per source.

Conditions:
  A45 / A50: siren, bark (near-field, INR +45/+50 dB) and urban hum all
             mutually separated by >= 50 deg (random placements)
  B45:       urban hum ALIGNED with the siren (known failure mode -- we
             quantify how often it costs us the siren)

Success criteria (per draw):
  DOA_siren / DOA_bark / DOA_urban : a candidate within 15 deg of the truth
  siren_recovered  : an output with f0 within 15 Hz of a siren harmonic AND
                     purity margins > 10 dB vs bark and vs urban
  bark_emitted     : a broadband output within 15 deg of the bark azimuth
  urban_attributed : the urban-marked output with the best margin whose beam
                     is within 15 deg of the urban azimuth (separated only)
  spurious_count   : outputs matching no source (bad angle and no f0 match)
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
    POS, FS, DUR, C, NFFT, far_pos, frac_delay, subspace_null,
)
from experiments.benchmark_blind_extraction import (
    track_f0, comb, notch_lines, full_path, lcmv_null_others,
    music_candidates, band_power_db, SIREN_LINES, URBAN_LINES, BARK_BANDS,
)

M = POS.shape[0]
SIREN_F0 = 350.0
N_DRAWS = 10
CONDITIONS = [("A45", 45.0, "sep"), ("A50", 50.0, "sep"), ("B45", 45.0, "ali")]


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def make_scene(seed, az_s, az_b, az_u, inr_db):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * DUR)) / FS
    r = np.random.default_rng(seed + 1)
    env = 0.5 + 0.5 * np.clip(0.8 * np.sin(2 * np.pi * 0.9 * t + r.uniform(0, 6)), -1, 1)
    siren = sum(np.sin(2 * np.pi * f0 * t + r.uniform(0, 6)) for f0 in SIREN_LINES) / 3.0 * env
    siren *= 10 ** (-20 / 20)

    bark_src = np.zeros(len(t))
    b, a = spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)], btype="band")
    for start in r.uniform(0.2, DUR - 0.4, size=6):
        n = int(0.12 * FS)
        i0 = int(start * FS)
        bark_src[i0:i0 + min(n, len(t) - i0)] += rng.standard_normal(min(n, len(t) - i0))
    bark_src = spsig.filtfilt(b, a, bark_src)

    urban_src = spsig.filtfilt(*spsig.butter(4, [250 / (FS / 2), 1500 / (FS / 2)], btype="band"),
                               [0.6 * rng.standard_normal(len(t))][0])
    urban_src = urban_src + 0.8 * np.sin(2 * np.pi * 430 * t + r.uniform(0, 6)) \
        + 0.5 * np.sin(2 * np.pi * 860 * t + r.uniform(0, 6))

    def prop_far(sig, az):
        u = far_pos(az) / 2000.0
        return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])

    def prop_near(sig, az, dist=10.0):
        d = np.linalg.norm(POS - far_pos(az, dist), axis=1)
        x = np.stack([frac_delay(sig, d[m] / C) for m in range(M)])
        return x  # external multiplier sets the true per-mic INR (no 1/r)

    x_s = prop_far(siren, az_s)
    x_b = prop_near(bark_src, az_b) * 10 ** (inr_db / 20) * 0.1 / np.sqrt(np.mean(bark_src ** 2) + 1e-12)
    x_u = prop_far(urban_src, az_u) * 10 ** (25 / 20) * 0.1 / np.sqrt(np.mean(urban_src ** 2) + 1e-12)
    noise = 0.1 * rng.standard_normal((M, len(t)))
    return x_s + x_b + x_u + noise, siren, bark_src, urban_src


def run_blind_once(mix):
    das = StandardDAS(POS, FS)
    cands = music_candidates(mix)
    outputs = []  # (az, f0_or_None, y, kind)
    raw_of = {}   # az -> chosen-raw, for global broadband pass
    for az in cands:
        others = [c for c in cands if ang(c, az) > 15]
        raws = {"DAS": das.process(mix, far_pos(az)),
                "SubspaceNull": subspace_null(mix, az, k=2),
                "LCMVnull": lcmv_null_others(mix, az, others)}
        paths = {n: full_path(y) for n, y in raws.items()}
        chosen = max(paths, key=lambda n: paths[n][2])
        f0_a, y_a, _, used_a = paths[chosen]
        # broadband ranking must use DAS raw: the full-path winner can be
        # SubspaceNull, which removes a dominant broadband source at its own beam
        raw_of[az] = (raws['DAS'], [f0_a] if f0_a else [])
        if f0_a is not None:
            outputs.append((az, f0_a, y_a, "tonal"))
            y_res = notch_lines(raws[chosen], f0_a)
            f0_b, score_b = track_f0(y_res)
            if f0_b is not None and score_b > 3.0 and abs(f0_b - f0_a) > 20:
                outputs.append((az, f0_b, comb(y_res, f0_b), "tonal"))
        else:
            outputs.append((az, None, y_a, "raw"))

    # Global broadband emission: rank beams by ABSOLUTE energy surviving the
    # notch of that beam's found tonal lines. A near-field bark dominates its
    # own beam's surviving energy far beyond leakage in other beams, so the
    # argmax beam identifies the broadband source without any truth.
    all_f0s = sorted({o[1] for o in outputs if o[1]})
    surviving = []
    for az, (y_raw, f0s) in raw_of.items():
        y_bb = y_raw
        for f0_n in all_f0s:
            y_bb = notch_lines(y_bb, f0_n)
        surviving.append((float(np.sum(y_bb ** 2)), az, y_bb))
    surviving.sort(reverse=True)
    if surviving and surviving[0][0] > 0:
        e_max, az_bb, y_bb = surviving[0]
        # emit top beams whose surviving energy is within 10 dB of the max
        outputs += [(az, None, y, "broadband")
                    for e, az, y in surviving if e > e_max / 10.0]
    return cands, outputs


def score_draw(cands, outputs, az_s, az_b, az_u, urban_separated):
    row = dict(DOA_siren=int(any(ang(c, az_s) < 15 for c in cands)),
               DOA_bark=int(any(ang(c, az_b) < 15 for c in cands)),
               DOA_urban=int(any(ang(c, az_u) < 15 for c in cands)))
    siren_best = None
    for az, f0, y, kind in outputs:
        if f0 is None:
            continue
        if not any(abs(f0 - SIREN_F0 * h) < 15 for h in (1, 2, 3)):
            continue
        m_b = band_power_db(y, freq_list=SIREN_LINES) - band_power_db(y, band_list=BARK_BANDS)
        m_u = band_power_db(y, freq_list=SIREN_LINES) - band_power_db(y, freq_list=URBAN_LINES)
        siren_best = (min(m_b, m_u), az, f0)
    row["siren_recovered"] = int(siren_best is not None and siren_best[0] > 10.0)
    row["siren_margin"] = siren_best[0] if siren_best else np.nan
    row["bark_emitted"] = int(any(kind == "broadband" and ang(az, az_b) < 15
                                  for az, _, _, kind in outputs))
    best_urban = None
    for az, f0, y, kind in outputs:
        if f0 is None:
            continue
        if not any(abs(f0 - 430.0 * h) < 15 for h in (1, 2)):
            continue
        m = band_power_db(y, freq_list=URBAN_LINES) - band_power_db(y, freq_list=SIREN_LINES)
        if best_urban is None or m > best_urban[0]:
            best_urban = (m, az)
    row["urban_attributed"] = int(urban_separated and best_urban is not None
                                  and ang(best_urban[1], az_u) < 15)
    matched = 0
    for az, f0, y, kind in outputs:
        if any(ang(az, a) < 15 for a in (az_s, az_b, az_u)):
            matched += 1
        elif f0 is not None and (any(abs(f0 - SIREN_F0 * h) < 15 for h in (1, 2, 3))
                                 or any(abs(f0 - 430.0 * h) < 15 for h in (1, 2))):
            matched += 1
    row["spurious"] = len(outputs) - matched
    return row


def main():
    rows = []
    for cond, inr, layout in CONDITIONS:
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(9000 + draw)
            az_s = float(rng.uniform(-170, 170))
            off = float(rng.choice([-1, 1]) * rng.uniform(60, 150))
            az_b = (az_s + off) % 360 - 360 * ((az_s + off) % 360 > 180)
            if layout == "sep":
                while True:
                    az_u = float(rng.uniform(-170, 170))
                    if ang(az_u, az_s) > 40 and ang(az_u, az_b) > 40:
                        break
            else:
                az_u = az_s
            seed = 9000 + draw + 100 * (0 if layout == "sep" else 1) + int(inr)
            mix, siren, bark, urban = make_scene(seed, az_s, az_b, az_u, inr)
            cands, outputs = run_blind_once(mix)
            row = score_draw(cands, outputs, az_s, az_b, az_u, layout == "sep")
            row.update(Condition=cond, Draw=draw, AzS=round(az_s), AzB=round(az_b), AzU=round(az_u))
            rows.append(row)
            print(f"[{cond}] draw {draw}: sirenDOA={row['DOA_siren']} barkDOA={row['DOA_bark']} "
                  f"urbanDOA={row['DOA_urban']} sirenOK={row['siren_recovered']} "
                  f"(margin {row['siren_margin']:.1f} dB) barkOUT={row['bark_emitted']} "
                  f"urbanATTR={row['urban_attributed']} spur={row['spurious']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "blind_mc_results.csv"), index=False)

    print("\n" + "=" * 92)
    print(f"BLIND PIPELINE MC ({N_DRAWS} draws/condition, randomized geometry/envelopes/INR)")
    print("=" * 92)
    for cond in df.Condition.unique():
        sub = df[df.Condition == cond]
        n = len(sub)

        def wilson(k):
            p = k / n
            z = 1.96
            lo = (p + z * z / (2 * n) - z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)
            hi = (p + z * z / (2 * n) + z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)
            return p, lo, hi

        print(f"\n--- {cond} ---")
        for key in ("DOA_siren", "DOA_bark", "DOA_urban", "siren_recovered",
                    "bark_emitted", "urban_attributed"):
            k = int(sub[key].sum())
            p, lo, hi = wilson(k)
            print(f"  {key:17s} {k}/{n}  = {p:5.0%}  (95% CI {lo:.0%}-{hi:.0%})")
        m = sub["siren_margin"].dropna()
        if len(m):
            print(f"  siren margin dB: median={m.median():.1f}  IQR=[{m.quantile(.25):.1f}, {m.quantile(.75):.1f}]")
        print(f"  spurious outputs: median={sub['spurious'].median():.0f}")


if __name__ == "__main__":
    main()

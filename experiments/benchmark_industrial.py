"""
Industrial robustness benchmark: NO interferer priors, NO gating.

The user's critique: burstiness-gating assumes intermittent interferers.
A continuous machine (+50 dB), a moving car, wind/rain, and channel
miscalibration (gain AND phase/delay) must all be handled by priors-free
spatial/spectral processing only.

Priors allowed: target-class only (tonal signature, known band 325-375 Hz
for localization - in the blind pipeline the f0 comes from the spectral
recovery stage).

Conditions (5 draws each):
  C50        continuous broadband machine +50 dB (gating is a no-op here)
  M45        MOVING broadband car +45 dB (azimuth sweeps ~35 deg in 3 s)
  C50W       C50 + wind phase decoherence
  each x {cal, miscal: gain +/-3 dB + per-mic delay error +/-1 cm}

Localizers compared (all UNGATED; gated MUSIC shown only to demonstrate
its failure on continuous interference):
  MUSIC_wide      NormMUSIC 250-1100 Hz
  MUSIC_nb        NormMUSIC 325-375 Hz (target line band)
  SRP_nb          SRP-PHAT 325-375 Hz (PHAT whitening = level-agnostic)
  SubMUSIC_nb     MUSIC_nb on the mixture with the top-1 spatial
                  component per bin projected out (prior-free removal of
                  whatever is loudest)
  GATED_MUSIC_nb  quietest-50%-frames MUSIC (prior-based, expected to fail)

Extraction (at oracle siren az, isolating localization): DAS vs
SubspaceNull(k=2) vs LCMV-null-others + comb at 350; margin vs machine
band and hum lines.
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
    POS, FS, DUR, C, NFFT, HOP, far_pos, frac_delay, steering_far, stft, istft,
)
from experiments.benchmark_blind_extraction import (
    comb, band_power_db, SIREN_LINES, URBAN_LINES,
)
from experiments.benchmark_blind_mc import ang

M = POS.shape[0]
SIREN_F0 = 350.0
BARK_BANDS = [(1550, 1750), (2100, 2300), (2600, 2800)]  # machine band proxy


def prop_far(sig, az):
    u = far_pos(az) / 2000.0
    return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])


def make_scene(seed, az_s, az_m, az_u, mode, wind):
    """mode: 'static' (machine at az_m) or 'moving' (car sweeping +-17.5 deg)."""
    rng = np.random.default_rng(seed)
    r = np.random.default_rng(seed + 1)
    t = np.arange(int(FS * DUR)) / FS
    env = 0.5 + 0.5 * np.clip(0.8 * np.sin(2 * np.pi * 0.9 * t + r.uniform(0, 6)), -1, 1)
    siren = sum(np.sin(2 * np.pi * f0 * t + r.uniform(0, 6)) for f0 in SIREN_LINES) / 3.0 * env
    siren *= 10 ** (-20 / 20)

    bb, aa = spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)], btype="band")
    machine_src = spsig.filtfilt(bb, aa, rng.standard_normal(len(t)))  # CONTINUOUS
    urban_src = spsig.filtfilt(*spsig.butter(4, [250 / (FS / 2), 1500 / (FS / 2)],
                                             btype="band"),
                               0.6 * rng.standard_normal(len(t)))
    urban_src += 0.8 * np.sin(2 * np.pi * 430 * t) + 0.5 * np.sin(2 * np.pi * 860 * t)

    x_s = prop_far(siren, az_s)
    if mode == "static":
        x_m = prop_far(machine_src, az_m)
    else:
        x_m = np.zeros((M, len(t)))
        block = 256
        src_idx = np.arange(len(t))
        for b0 in range(0, len(t) - block, block):
            az_t = az_m + 17.5 * np.sin(2 * np.pi * (b0 / FS) / DUR)
            u = far_pos(az_t) / 2000.0
            tau = -(POS @ u) / C
            for m in range(M):
                pos = (np.arange(b0, b0 + block) / FS) * -1.0 + tau[m] * -1.0
                pos = (np.arange(b0, b0 + block) / FS) + tau[m]
                ok = (pos >= 0) & (pos < len(t) - 1)
                seg = np.zeros(block)
                seg[ok] = np.interp(pos[ok], src_idx, machine_src)
                x_m[m, b0:b0 + block] = seg
    inr = 50.0 if mode == "static" else 45.0
    x_m *= 10 ** (inr / 20) * 0.1 / np.sqrt(np.mean(machine_src ** 2) + 1e-12)
    x_u = prop_far(urban_src, az_u) * 10 ** (25 / 20) * 0.1 / np.sqrt(np.mean(urban_src ** 2) + 1e-12)

    if wind:
        freqs = np.fft.rfftfreq(NFFT, 1 / FS)
        freqs[0] = 1e-6
        from simulator.acoustics.channel import apply_turbulence_decoherence
        for x, az in ((x_s, az_s), (x_m, az_m), (x_u, az_u)):
            Z = stft(x)
            Z = apply_turbulence_decoherence(Z, POS, far_pos(az), freqs, sigma_phi_base=0.12)
            x[:] = istft(Z)[:, :x.shape[1]]
    mix = x_s + x_m + x_u + 0.1 * rng.standard_normal((M, len(t)))
    return mix


def apply_miscal(mix, seed):
    """Per-mic gain +/-3 dB + delay error +/-1 cm (phase slope) - channel
    non-idealities an uncalibrated installation actually has."""
    rng = np.random.default_rng(seed + 777)
    out = np.zeros_like(mix)
    for m in range(M):
        g = 10 ** (rng.uniform(-3, 3) / 20)
        tau_err = rng.uniform(-0.01, 0.01) / C
        out[m] = frac_delay(mix[m], tau_err) * g
    return out


def localizers(mix, az_s):
    Z = spsig.stft(mix, fs=FS, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")[2]
    az_grid = np.deg2rad(np.arange(-180, 180, 2.0))
    out = {}
    for name, cls, fr in [("MUSIC_wide", pra.doa.NormMUSIC, (250., 1100.)),
                          ("MUSIC_nb", pra.doa.NormMUSIC, (325., 375.)),
                          ("SRP_nb", pra.doa.SRP, (325., 375.))]:
        kw = {"nfft": NFFT, "mode": "far"} if name == "SRP_nb" else {"nfft": NFFT}
        d = cls(POS.T.copy(), FS, azimuth=az_grid, **kw)
        d.locate_sources(Z, num_src=3, freq_range=list(fr))
        est = np.rad2deg(np.atleast_1d(d.azimuth_recon))
        out[name] = int(any(ang(e, az_s) < 15 for e in est))
    # subspace-removed narrowband MUSIC: project out top-1 spatial component
    _, F, T = Z.shape
    Zp = np.zeros_like(Z)
    for f in range(F):
        X = Z[:, f, :]
        R = (X @ X.conj().T) / T
        evals, evecs = np.linalg.eigh(R)
        v = evecs[:, -1]
        Zp[:, f, :] = X - np.outer(v, v.conj() @ X)
    d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=NFFT, azimuth=az_grid)
    d.locate_sources(Zp, num_src=3, freq_range=[325., 375.])
    est = np.rad2deg(np.atleast_1d(d.azimuth_recon))
    out["SubMUSIC_nb"] = int(any(ang(e, az_s) < 15 for e in est))
    # gated variant (prior-based; no-op for continuous interference)
    fe = np.sum(np.abs(Z[0]) ** 2, axis=0)
    Zq = Z[:, :, fe < np.percentile(fe, 50)]
    d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=NFFT, azimuth=az_grid)
    d.locate_sources(Zq, num_src=3, freq_range=[325., 375.])
    est = np.rad2deg(np.atleast_1d(d.azimuth_recon))
    out["GATED_MUSIC_nb"] = int(any(ang(e, az_s) < 15 for e in est))
    return out


def extraction_margin(mix, az_s):
    from experiments.benchmark_extreme_inr import subspace_null
    das = StandardDAS(POS, FS)
    y_das = das.process(mix, far_pos(az_s))
    y_sub = subspace_null(mix, az_s, k=2)
    best = None
    for y in (y_das, y_sub):
        yc = comb(y, SIREN_F0)
        m = (band_power_db(yc, freq_list=SIREN_LINES)
             - max(band_power_db(yc, band_list=BARK_BANDS),
                   band_power_db(yc, freq_list=URBAN_LINES)))
        if best is None or m > best:
            best = m
    return best


def main():
    rows = []
    conds = [("C50", "static", False), ("M45", "moving", False), ("C50W", "static", True)]
    for cond, mode, wind in conds:
        for cal in ("cal", "miscal"):
            for draw in range(5):
                rng = np.random.default_rng(7000 + draw)
                az_s = float(rng.uniform(-170, 170))
                off = float(rng.choice([-1, 1]) * rng.uniform(60, 150))
                az_m = (az_s + off) % 360 - 360 * ((az_s + off) % 360 > 180)
                while True:
                    az_u = float(rng.uniform(-170, 170))
                    if ang(az_u, az_s) > 40 and ang(az_u, az_m) > 40:
                        break
                seed = 7000 + draw + 1000 * (0 if cal == "cal" else 1)
                mix = make_scene(seed, az_s, az_m, az_u, mode, wind)
                if cal == "miscal":
                    mix = apply_miscal(mix, seed)
                loc = localizers(mix, az_s)
                margin = extraction_margin(mix, az_s)
                rows.append(dict(Cond=cond, Cal=cal, Draw=draw, **loc, Margin=margin))
                print(f"[{cond}/{cal}] draw {draw}: " +
                      " ".join(f"{k}={v}" for k, v in loc.items()) +
                      f"  margin={margin:.1f} dB", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "industrial_benchmark.csv"), index=False)
    print("\n" + "=" * 96)
    print("INDUSTRIAL (no gating, no interferer priors): siren-DOA hit rate + extraction margin")
    print("=" * 96)
    g = df.groupby(["Cond", "Cal"])
    summ = g.agg({k: "mean" for k in ("MUSIC_wide", "MUSIC_nb", "SRP_nb",
                                      "SubMUSIC_nb", "GATED_MUSIC_nb")})
    summ["Margin_med"] = g["Margin"].median()
    print(summ.to_string(float_format=lambda v: f"{v:6.2f}"))


if __name__ == "__main__":
    main()

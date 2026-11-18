"""
Unified benchmark with the corrected architecture:

  Stage A - DOA estimators  -> ANGLES ONLY (pyroomacoustics SRP / NormMUSIC / FRIDA)
  Stage B - Grid beamformers -> directional audio + spatial map (peak, SSR depth)
  Stage C - Gridless BSS     -> ALL separated sources + one DOA per source
                                (pra.bss.auxiva + projection-back manifold fit)
  Stage D - Target identification -> repo's harmonic feature extraction selects
                                which source/beam is the target (no LAM argmax)

Scene: 8-mic circular array (alias-free below 2.8 kHz), 3 far-field sources:
  target   az +40 deg : harmonic stack 420..1680 Hz (drone-like)
  interf.1 az -60 deg : bandpass noise 300..2000 Hz, +4 dB vs target
  interf.2 az +150 deg: bandpass noise 500..1500 Hz, +1 dB vs target
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

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from experiments.feature_extraction.signal_processing import compute_spectrogram_and_peaks
from experiments.feature_extraction.harmonic_detection import track_harmonics

FS = 16000
DUR = 4.0
M = 8
R_ARRAY = 0.08
C = 343.0
TRUE_DOAS = [40.0, -60.0, 150.0]      # [target, interf1, interf2]
TARGET_AZ = TRUE_DOAS[0]

rng = np.random.default_rng(7)

_ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
POS = np.stack([R_ARRAY * np.cos(_ang), R_ARRAY * np.sin(_ang), np.zeros(M)], axis=1)


def src_pos(az_deg, dist=120.0):
    a = np.deg2rad(az_deg)
    return dist * np.array([np.cos(a), np.sin(a), 0.0])


def frac_delay(sig, tau):
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1.0 / FS)
    return np.fft.irfft(S * np.exp(-2j * np.pi * f * tau), n=len(sig))


def bandpass_noise(lo, hi, n):
    x = rng.standard_normal(n)
    b, a = spsig.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return spsig.filtfilt(b, a, x)


def slow_env(seed, freqs=(0.7, 1.3, 2.1)):
    """Slow random amplitude modulation: IVA/BSS identifies sources through
    their distinct nonstationary envelopes, so stationary tones/noise would
    make the problem unidentifiable (and is unrealistic for drones anyway)."""
    r = np.random.default_rng(seed)
    t = np.arange(int(FS * DUR)) / FS
    e = np.zeros_like(t)
    for f in freqs:
        e += np.sin(2 * np.pi * f * t + r.uniform(0, 6.28)) * r.uniform(0.5, 1.5)
    return 0.5 + 0.5 * np.clip(e / 3, -1, 1)


def make_scene(gain_spread_db=0.0, return_refs=False, wind=False, drift=False):
    """gain_spread_db: total per-mic gain spread (uniform in +/-(spread/2) dB),
    simulating uncalibrated hardware (e.g. 6 dB spread = each mic +/-3 dB).
    wind: apply turbulence phase decoherence to each source contribution.
    drift: slow +/-2% RPM wander on the target's harmonic stack."""
    t = np.arange(int(FS * DUR)) / FS
    if drift:
        scale = 1.0 + 0.02 * np.sin(2 * np.pi * 0.23 * t + 1.1)
        ph = np.cumsum(2 * np.pi * np.outer(np.array([420, 840, 1260, 1680]),
                                            scale) / FS, axis=1)
        target = np.sum(np.sin(ph) / 4.0, axis=0) * slow_env(1)
    else:
        target = sum(np.sin(2 * np.pi * h * t) for h in [420, 840, 1260, 1680]) / 4.0 * slow_env(1)
    int1 = bandpass_noise(300, 2000, len(t)) * slow_env(2)
    int2 = bandpass_noise(500, 1500, len(t)) * slow_env(3)

    def prop(sig, az):
        p = src_pos(az)
        out = np.stack([frac_delay(sig, np.linalg.norm(POS[m] - p) / C) for m in range(M)])
        return out / np.sqrt(np.mean(out**2))

    def wind_jitter(x, az, sigma=0.12):
        from simulator.acoustics.channel import apply_turbulence_decoherence
        Z = spsig.stft(x, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
        freqs = np.fft.rfftfreq(512, 1 / FS)
        freqs[0] = 1e-6
        Zp = apply_turbulence_decoherence(Z, POS, src_pos(az), freqs, sigma_phi_base=sigma)
        _, xw = spsig.istft(Zp, fs=FS, nperseg=512, noverlap=256, window="hann")
        return xw

    x_t = prop(target, TARGET_AZ)
    x_1 = prop(int1, -60.0) * 10 ** (4.0 / 20)
    x_2 = prop(int2, 150.0) * 10 ** (1.0 / 20)
    if wind:
        x_t = wind_jitter(x_t, TARGET_AZ)
        x_1 = wind_jitter(x_1, -60.0)
        x_2 = wind_jitter(x_2, 150.0)
    noise = 0.1 * rng.standard_normal((M, len(t)))
    mix = x_t + x_1 + x_2 + noise
    if gain_spread_db > 0:
        g_rng = np.random.default_rng(1000 + int(gain_spread_db * 10))
        gains_db = g_rng.uniform(-gain_spread_db / 2.0, gain_spread_db / 2.0, size=(M, 1))
        mix = mix * 10 ** (gains_db / 20.0)
    if return_refs:
        return mix, target, int1, int2
    return mix, target


def steering(az_deg, freqs):
    """Fixed convention: a = exp(-j w tau), tau = -(p.u)/c  -> (F, M)."""
    u = src_pos(az_deg)
    u = u / np.linalg.norm(u)
    tau = -(POS @ u) / C
    return np.exp(-1j * 2 * np.pi * np.outer(freqs, tau)) / np.sqrt(M)


def harmonic_score(audio):
    """Repo's harmonic feature extractor -- used ONLY for relative target ID."""
    f, t, Pxx, peaks = compute_spectrogram_and_peaks(np.asarray(audio), FS, parallel=False)
    tracks = track_harmonics(peaks)
    return float(tracks[0]["max_score"]) if tracks else -150.0


def si_sdr_aligned(y, ref):
    """SI-SDR after best cross-correlation lag alignment (removes common delay)."""
    y = np.asarray(y).flatten()
    ref = np.asarray(ref).flatten()
    n = min(len(y), len(ref))
    corr = spsig.correlate(y[:n] - y[:n].mean(), ref[:n] - ref[:n].mean(),
                           mode="full", method="fft")
    lag = np.argmax(np.abs(corr)) - (n - 1)
    if lag > 0:
        y_al, r_al = y[lag:], ref[: len(y) - lag]
    else:
        y_al, r_al = y[: len(y) + lag], ref[-lag:]
    m = min(len(y_al), len(r_al))
    y_al, r_al = y_al[:m], r_al[:m]
    alpha = np.dot(y_al, r_al) / (np.dot(y_al, y_al) + 1e-12)
    err = alpha * y_al - r_al
    return float(10 * np.log10(np.dot(r_al, r_al) / (np.dot(err, err) + 1e-12)))


def angular_err(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def match_doas(estimates, truths):
    """Greedy match; returns per-truth error list and spurious count."""
    est = sorted(estimates, key=lambda e: min(angular_err(e, t) for t in truths))
    used, errs = set(), []
    for t in truths:
        best, bi = 1e9, -1
        for i, e in enumerate(est):
            if i in used:
                continue
            d = angular_err(e, t)
            if d < best:
                best, bi = d, i
        used.add(bi)
        errs.append(best)
    return errs, max(0, len(est) - len(truths))


# ---------------------------------------------------------------- Stage A
def stage_a_doa(mix, cond="clean"):
    print("\n" + "=" * 78)
    print("STAGE A -- DOA ESTIMATORS (output: angles only, no audio)")
    print("=" * 78)
    Z = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]  # (M,F,T)
    L = POS.T.copy()
    az_grid = np.deg2rad(np.arange(-180, 180, 1.0))
    rows = []
    for name in ("SRP", "NormMUSIC", "FRIDA"):
        try:
            cls = getattr(pra.doa, name)
            kw = {"nfft": 512, "mode": "far"} if name == "SRP" else {"nfft": 512}
            try:
                d = cls(L, FS, azimuth=az_grid, **kw)
            except TypeError:
                d = cls(L, FS, **kw)
            t0 = time.perf_counter()
            d.locate_sources(Z, num_src=len(TRUE_DOAS), freq_range=[300.0, 2000.0])
            dt = time.perf_counter() - t0
            est = np.rad2deg(np.atleast_1d(d.azimuth_recon)).tolist()
            note = ""
        except AssertionError:
            # FRIDA's FRI basis needs M >= K; the fine azimuth grid and
            # multi-source setting both inflate K beyond M=8. Rebuild with
            # FRIDA's default grid and single-source mode (fine on a 33+ mic array).
            d = pra.doa.FRIDA(L, FS, nfft=512)
            d.locate_sources(Z)
            dt = time.perf_counter() - t0
            est = np.rad2deg(np.atleast_1d(d.azimuth_recon)).tolist()
            note = " [single-src fallback: M=8 too small for multi-source FRI]"
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
            continue
        errs, spur = match_doas(est, TRUE_DOAS)
        for truth, e in zip(TRUE_DOAS, errs):
            rows.append({"Condition": cond, "Stage": "A_DOA", "Method": name, "Truth_az": truth,
                         "DOA_err_deg": e, "Spurious": spur, "RTF": dt / DUR})
        print(f"  {name:10s} estimates={np.round(est,1)}  errors={np.round(errs,1)} deg  "
              f"spurious={spur}  rtf={dt/DUR:.3f}{note}")
    return rows


# ---------------------------------------------------------------- Stage B
def stage_b_grid(mix, ref, cond="clean"):
    print("\n" + "=" * 78)
    print("STAGE B -- GRID BEAMFORMERS (output: directional audio + map)")
    print("=" * 78)
    azs = np.arange(-180, 180, 5.0)
    rows = []
    for name, cls in [("StandardDAS", StandardDAS), ("LCMVSMI", LCMVSMI),
                      ("RobustLCMV", RobustLCMV), ("PhaseOnlyLCMV", PhaseOnlyLCMV)]:
        bf = cls(POS, FS)
        t0 = time.perf_counter()
        powers, audios = [], {}
        for az in azs:
            y = bf.process(mix, src_pos(az))
            powers.append(float(np.mean(y**2)))
            audios[az] = y
        sweep_t = time.perf_counter() - t0
        powers = np.array(powers)
        peak_az = float(azs[np.argmax(powers)])
        ssr = 10 * np.log10(np.max(powers) / max(np.min(powers), 1e-30))
        y_oracle = audios[TARGET_AZ if TARGET_AZ in audios else azs[np.argmin(np.abs(azs - TARGET_AZ))]]
        sisdr_or = si_sdr_aligned(y_oracle, ref)
        sisdr_pk = si_sdr_aligned(audios[peak_az], ref)

        # top-3 local peaks -> harmonic-feature target selection (mini pipeline B)
        idx_sorted = np.argsort(powers)[::-1]
        top3, taken = [], []
        for i in idx_sorted:
            if all(abs(angular_err(azs[i], t)) > 15 for t in taken):
                top3.append(i)
                taken.append(azs[i])
            if len(top3) == 3:
                break
        cand = [(azs[i], harmonic_score(audios[azs[i]])) for i in top3]
        sel_az = max(cand, key=lambda c: c[1])[0]

        rows += [
            {"Condition": cond, "Stage": "B_Grid", "Method": name, "Truth_az": TARGET_AZ,
             "Peak_az": peak_az, "Peak_err_deg": angular_err(peak_az, TARGET_AZ),
             "SSR_depth_dB": ssr, "SI-SDR_oracle_dB": sisdr_or,
             "SI-SDR_energyPeak_dB": sisdr_pk, "SelAz_featID": sel_az,
             "SelErr_featID_deg": angular_err(sel_az, TARGET_AZ), "RTF": sweep_t / DUR},
        ]
        print(f"  {name:14s} energyPeak@{peak_az:6.1f} (err {angular_err(peak_az,TARGET_AZ):5.1f})  "
              f"SSR={ssr:5.1f} dB  SI-SDR oracle={sisdr_or:6.1f}  featID picks {sel_az:6.1f} "
              f"(err {angular_err(sel_az,TARGET_AZ):5.1f})  rtf={sweep_t/DUR:.2f}")
    return rows


# ---------------------------------------------------------------- Stage C
def manifold_doa(Z_x, y_k, freqs, az_grid):
    """Projection-back steering estimate per bin -> best manifold match."""
    _, F, T = Z_x.shape
    Zy = spsig.stft(y_k, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
    if Zy.ndim == 3:  # (1, F, T) when input is 2-D
        Zy = Zy[0]  # (F,T)
    e_f = np.sum(np.abs(Zy) ** 2, axis=1)
    bins = np.argsort(e_f)[::-1][:40]
    bins = bins[freqs[bins] > 100]
    resp = np.zeros(len(az_grid))
    a_grid = [steering(az, freqs) for az in az_grid]
    for f in bins:
        X = Z_x[:, f, :]
        R = (X @ X.conj().T) / T + 1e-4 * np.eye(M)
        r = (X @ Zy[f].conj()) / T
        a_est = np.linalg.solve(R, r)
        a_est /= (np.linalg.norm(a_est) + 1e-12)
        for gi, a in enumerate(a_grid):
            af = a[f]
            resp[gi] += np.abs(a_est @ af.conj()) ** 2  # both unit-norm
    return float(az_grid[int(np.argmax(resp))])


def stage_c_bss(mix, ref, cond="clean"):
    print("\n" + "=" * 78)
    print("STAGE C -- GRIDLESS BSS, MULTI-OUTPUT + DOA PER SOURCE (pra.bss.auxiva)")
    print("=" * 78)
    rows = []
    Z = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]  # (M,F,T)
    X_stft = np.transpose(Z, (2, 1, 0))  # (T,F,M) as required by pra.bss.auxiva
    t0 = time.perf_counter()
    try:
        Y = pra.bss.auxiva(X_stft, n_src=len(TRUE_DOAS), n_iter=50, proj_back=True)
        ys = []
        for k in range(Y.shape[2]):
            _, y_k = spsig.istft(Y[:, :, k].T, fs=FS, nperseg=512, noverlap=256, window="hann")
            ys.append(y_k)
        y = np.stack(ys)
        mode = f"overdetermined (M={M} -> K={len(TRUE_DOAS)})"
    except Exception as e:
        print(f"  auxiva overdetermined failed ({e}); falling back to 3-mic subset")
        sel = [0, 3, 5]
        Zs = spsig.stft(mix[sel], fs=FS, nperseg=512, noverlap=256, window="hann")[2]
        Y = pra.bss.auxiva(np.transpose(Zs, (2, 1, 0)), n_src=len(TRUE_DOAS),
                           n_iter=30, proj_back=True)
        ys = []
        for k in range(Y.shape[2]):
            _, y_k = spsig.istft(Y[:, :, k].T, fs=FS, nperseg=512, noverlap=256, window="hann")
            ys.append(y_k)
        y = np.stack(ys)
        mode = "determined subset (3 mics)"
    sep_t = time.perf_counter() - t0

    freqs = np.fft.rfftfreq(512, 1 / FS)
    Z_x = spsig.stft(mix, fs=FS, nperseg=512, noverlap=256, window="hann")[2]
    az_grid = np.arange(-180, 180, 2.0)

    scores = []
    for k in range(y.shape[0]):
        t1 = time.perf_counter()
        doa = manifold_doa(Z_x, y[k], freqs, az_grid)
        fit_t = time.perf_counter() - t1
        hs = harmonic_score(y[k])
        sisdr = si_sdr_aligned(y[k], ref)
        err = angular_err(doa, TARGET_AZ)
        scores.append(hs)
        rows.append({"Condition": cond, "Stage": "C_BSS", "Method": f"AuxIVA_src{k+1}", "Truth_az": TARGET_AZ,
                     "DOA_est": doa, "DOA_err_vs_target": err, "Harmonic_score": hs,
                     "SI-SDR_dB": sisdr, "RTF": (sep_t / 3 + fit_t) / DUR})
        print(f"  source {k+1}: DOA={doa:7.1f} deg  err_vs_target={err:6.1f}  "
              f"harmonic={hs:7.1f}  SI-SDR={sisdr:6.1f} dB")

    k_sel = int(np.argmax(scores))
    sel_row = rows[k_sel]
    print(f"  -> feature-extraction ID: source {k_sel+1} is the target "
          f"(highest harmonic score); reported DOA = {sel_row['DOA_est']:.1f} deg "
          f"(error {sel_row['DOA_err_vs_target']:.1f} deg)   [{mode}]")
    return rows


CONDITIONS = [(0.0, "clean"), (6.0, "gain_imbalance_6dB"), (12.0, "gain_imbalance_12dB")]


def main():
    print(f"Unified benchmark | M={M} circular r={R_ARRAY} m | fs={FS} | {DUR}s | "
          f"truth DOAs {TRUE_DOAS} (target={TARGET_AZ})")
    rows = []
    for spread, cond in CONDITIONS:
        print()
        print("#" * 78)
        print(f"# CONDITION: {cond}  (per-mic gains uniform in +/-{spread/2:g} dB)")
        print("#" * 78)
        mix, ref = make_scene(spread)
        rows += stage_a_doa(mix, cond)
        rows += stage_b_grid(mix, ref, cond)
        rows += stage_c_bss(mix, ref, cond)
    df = pd.DataFrame(rows)
    out = os.path.join(REPO, "docs", "unified_benchmark.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

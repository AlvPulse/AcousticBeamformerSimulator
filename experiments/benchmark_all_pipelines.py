"""
ALL-PIPELINES benchmark vs the company gold standard, on a scenario library.

Company gold standard (L0): grid LCMV scan -> energy peak -> audio@peak
-> (their history-anomaly clustering + detector downstream).

Incremental alternatives:
  L1a grid + PhaseOnly      (algorithm swap)
  L1b grid + RobustLCMV     (algorithm swap, fixed CMT)
  L1c grid + GSC + gain-norm(algorithm swap, best uncalibrated)
  L2  grid + DAS            (radical simplification)
  L3a DOA-first: NormMUSIC -> GSC+gn at candidates -> tonality selection
  L3b DOA-first: NormMUSIC -> SubspaceNull -> comb post-filter
  L4  BSS: OverIVA multi-output -> manifold DOA -> tonality selection

Scenario library (13): audio classes x interference regimes x calibration:
  S01 far-field tonal only (sanity)
  S02 + far broadband interferer +10 dB (moderate)
  S03 + far broadband +45 dB (extreme INR)
  S04 + NEAR-FIELD bark bursts +45 dB (intermittent extreme)
  S05 + CONTINUOUS machine +50 dB (static-continuous extreme)
  S06 + MOVING broadband +45 dB
  S07 + urban tonal hum +25 dB & broadband +15
  S08 two weak tonals (competing targets)
  S09 = S04 + gain imbalance 12 dB
  S10 = S04 + all-pass phase error +/-20 deg
  S11 = S05 + wind
  S12 BIRD-CHIRP target (sweeps; tonality selection should struggle)
  S13 machinery-AM target vs machine interferer (both broadband-ish)

Grid-scan selection for LCMV/PhaseOnly/RobustLCMV is computed exactly via
the Capon spectrum (argmax of the constrained output power over the grid
IS the Capon spectrum argmax), so scans are fast and faithful.

Gain-norm adversarial block: gains estimated on a NEAR-FIELD-interferer-
dominated scene, then applied to a far-field scene -- quantifies the
user's concern that scene-dependent 'calibration' bakes acoustics into
hardware gains. Estimators compared: naive RMS (near-field scene),
proper RMS (far-field-quiet scene), coherence-weighted.
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
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_extreme_inr import (
    POS, FS, DUR, C, NFFT, far_pos, frac_delay, steering_far, stft, subspace_null,
)
from experiments.benchmark_blind_extraction import (
    music_candidates, comb, band_power_db, track_f0,
)
from experiments.benchmark_phase_robustness import align, si_sdr

M = POS.shape[0]
AZ_T = 40.0
rng = np.random.default_rng(3)


# ------------------------------------------------- audio class generators
def gen_tonal(t, seed=0):
    r = np.random.default_rng(seed)
    env = 0.5 + 0.5 * np.clip(0.8 * np.sin(2 * np.pi * 0.9 * t + r.uniform(0, 6)), -1, 1)
    return sum(np.sin(2 * np.pi * f * t) for f in (350, 700, 1050)) / 3 * env


def gen_chirp(t):
    y = np.zeros_like(t)
    for i, start in enumerate(np.arange(0.2, DUR - 0.3, 0.35)):
        n = int(0.12 * FS)
        i0 = int(start * FS)
        tt = np.arange(min(n, len(t) - i0)) / FS
        f0, f1 = 1800 + 300 * (i % 3), 3200 + 300 * (i % 3)
        seg = np.sin(2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * 0.12)))
        y[i0:i0 + len(seg)] += seg * np.hanning(len(seg))
    return y


def gen_machinery(t, seed=1):
    r = np.random.default_rng(seed)
    tone = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    harsh = spsig.filtfilt(*spsig.butter(4, [200 / (FS / 2), 2500 / (FS / 2)],
                                         btype="band"), r.standard_normal(len(t)))
    return tone + 0.4 * harsh / np.sqrt(np.mean(harsh ** 2))


def gen_bark(t, seed=2):
    r = np.random.default_rng(seed)
    y = np.zeros_like(t)
    for start in r.uniform(0.2, DUR - 0.4, size=6):
        i0 = int(start * FS)
        n = min(int(0.12 * FS), len(t) - i0)
        y[i0:i0 + n] += r.standard_normal(n)
    return spsig.filtfilt(*spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)],
                                        btype="band"), y)


def gen_bb(t, seed=3, lo=300, hi=3000):
    r = np.random.default_rng(seed)
    return spsig.filtfilt(*spsig.butter(4, [lo / (FS / 2), hi / (FS / 2)],
                                        btype="band"), r.standard_normal(len(t)))


def gen_hum(t):
    return 0.8 * np.sin(2 * np.pi * 430 * t) + 0.5 * np.sin(2 * np.pi * 860 * t) \
        + 0.6 * gen_bb(t, seed=4, lo=250, hi=1500) / np.sqrt(np.mean(gen_bb(t, seed=4, lo=250, hi=1500) ** 2))


def prop_far(sig, az):
    u = far_pos(az) / 2000.0
    return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])


def prop_near(sig, az, dist=10.0):
    d = np.linalg.norm(POS - far_pos(az, dist), axis=1)
    return np.stack([frac_delay(sig, d[m] / C) for m in range(M)])


def scale_to(x, db_over_noise, noise_rms=0.1):
    return x * (db_over_noise * noise_rms) / np.sqrt(np.mean(x ** 2) + 1e-12)


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ------------------------------------------------- scenarios
def build_scenario(sid):
    t = np.arange(int(FS * DUR)) / FS
    noise = 0.1 * rng.standard_normal((M, len(t)))
    tgt_db, mix_extra = -10.0, None
    if sid == "S01":
        tgt = gen_tonal(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) / 1 + noise
    elif sid == "S02":
        tgt = gen_tonal(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_far(gen_bb(t), -60) * 10 ** ((tgt_db + 10) / 20) + noise
    elif sid == "S03":
        tgt = gen_tonal(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_far(gen_bb(t), -60) * 10 ** ((tgt_db + 45) / 20) + noise
    elif sid == "S04":
        tgt = gen_tonal(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_near(gen_bark(t), -60) * 10 ** ((tgt_db + 45) / 20) + noise
    elif sid == "S05":
        tgt = gen_tonal(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_far(gen_bb(t, seed=5), -60) * 10 ** ((tgt_db + 50) / 20) + noise
    elif sid == "S06":
        tgt = gen_tonal(t)
        x_m = np.zeros((M, len(t)))
        for b0 in range(0, len(t) - 256, 256):
            az_t = -60 + 17.5 * np.sin(2 * np.pi * (b0 / FS) / DUR)
            u = far_pos(az_t) / 2000.0
            tau = -(POS @ u) / C
            for m in range(M):
                pos = np.arange(b0, b0 + 256) / FS + tau[m]
                ok = (pos >= 0) & (pos < len(t) - 1)
                seg = np.zeros(256)
                seg[ok] = np.interp(pos[ok], np.arange(len(t)), gen_bb(t, seed=5))
                x_m[m, b0:b0 + 256] = seg
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + x_m * 10 ** ((tgt_db + 45) / 20) / np.sqrt(np.mean(gen_bb(t, seed=5) ** 2)) + noise
    elif sid == "S07":
        tgt = gen_tonal(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_far(gen_hum(t), 120) * 10 ** ((tgt_db + 25) / 20) + noise
    elif sid == "S08":
        tgt = gen_tonal(t)
        other = gen_tonal(t, seed=9)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_far(other, 150) * 10 ** ((tgt_db - 3) / 20) + noise
    elif sid == "S09":
        tgt = gen_tonal(t)
        base = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_near(gen_bark(t), -60) * 10 ** ((tgt_db + 45) / 20) + noise
        g = 10 ** (rng.uniform(-6, 6, (M, 1)) / 20)
        mix = base * g
    elif sid == "S10":
        tgt = gen_tonal(t)
        base = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_near(gen_bark(t), -60) * 10 ** ((tgt_db + 45) / 20) + noise
        Z = stft(base)
        ph = np.exp(1j * np.deg2rad(rng.uniform(-20, 20, (M, 1, 1))))
        mix = spsig.istft(Z * ph, fs=FS, nperseg=NFFT, noverlap=NFFT - 256,
                          window="hann")[1][:, :base.shape[1]]
    elif sid == "S11":
        from simulator.acoustics.channel import apply_turbulence_decoherence
        tgt = gen_tonal(t)
        x_s = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20)
        x_m = prop_far(gen_bb(t, seed=5), -60) * 10 ** ((tgt_db + 50) / 20)
        freqs = np.fft.rfftfreq(NFFT, 1 / FS)
        freqs[0] = 1e-6
        for x, az_ in ((x_s, AZ_T), (x_m, -60.0)):
            Zx = stft(x)
            x[:] = spsig.istft(apply_turbulence_decoherence(Zx, POS, far_pos(az_), freqs,
                                                            sigma_phi_base=0.12),
                               fs=FS, nperseg=NFFT, noverlap=NFFT - 256, window="hann")[1][:, :x.shape[1]]
        mix = x_s + x_m + noise
    elif sid == "S12":
        tgt = gen_chirp(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_near(gen_bark(t), -60) * 10 ** ((tgt_db + 45) / 20) + noise
    elif sid == "S13":
        tgt = gen_machinery(t)
        mix = prop_far(tgt, AZ_T) * 10 ** (tgt_db / 20) + prop_far(gen_bb(t, seed=6, lo=250, hi=2800), -60) * 10 ** ((tgt_db + 20) / 20) + noise
    return mix, tgt


# ------------------------------------------------- pipeline pieces
def gain_normalize(x):
    r = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))
    return x / (r + 1e-12)


def capon_map(mix, kind="lcmv", grid=None):
    """Exact grid-argmax surrogate for LCMV/PhaseOnly/RobustLCMV scans.
    (argmax over look directions of the constrained output power IS the
    Capon spectrum argmax.) One eigendecomposition per bin, reused."""
    Z = stft(mix)
    _, F, T = Z.shape
    if kind == "phaseonly":
        Z = Z / (np.abs(Z) + 1e-12)
    taper = RobustLCMV(POS, FS).T_cmt if kind == "robust" else None
    Rinv_list = []
    for f in range(F):
        X = Z[:, f, :]
        R = (X @ X.conj().T) / T
        if taper is not None:
            R = R * taper[f]
        ev, evec = np.linalg.eigh(R + 1e-3 * np.trace(R).real / M * np.eye(M))
        ev = np.maximum(ev, 1e-3 * float(np.max(ev)))
        Rinv_list.append((evec * (1 / ev)) @ evec.conj().T)
    azs = grid if grid is not None else np.arange(-180, 180, 10.0)
    pows = np.zeros(len(azs))
    A_cache = [steering_far(a) for a in azs]
    for i, A in enumerate(A_cache):
        tot = 0.0
        for f in range(F):
            a = A[f]
            tot += float(1.0 / np.real(a.conj() @ Rinv_list[f] @ a))
        pows[i] = tot
    return azs, pows


def tonality8k(y):
    f, P = spsig.periodogram(np.asarray(y).flatten(), FS, nfft=32768,
                             window="blackman")
    band = (f > 200) & (f < 3000)
    Pb = np.mean(P[:, None], axis=1) if P.ndim > 1 else P
    Pb = P[band]
    return float(10 * np.log10(np.max(Pb) / (np.median(Pb) + 1e-15)))


def extract_audio(pipeline, mix, sel_az):
    das = StandardDAS(POS, FS)
    if pipeline in ("L0_gridLCMV",):
        return LCMVSMI(POS, FS).process(mix, far_pos(sel_az))
    if pipeline == "L1a_gridPhaseOnly":
        return PhaseOnlyLCMV(POS, FS).process(mix, far_pos(sel_az))
    if pipeline == "L1b_gridRobustLCMV":
        return RobustLCMV(POS, FS).process(mix, far_pos(sel_az))
    if pipeline == "L1c_gridGSCgn":
        return GSC_MCWF(POS, FS).process(gain_normalize(mix), far_pos(sel_az))
    if pipeline == "L2_gridDAS":
        return das.process(mix, far_pos(sel_az))
    if pipeline == "L3a_music+GSCgn":
        return GSC_MCWF(POS, FS).process(gain_normalize(mix), far_pos(sel_az))
    if pipeline == "L3b_music+SubspaceNull+comb":
        return subspace_null(mix, sel_az, k=2)
    raise ValueError(pipeline)


def run_pipeline(name, mix, ref):
    das = StandardDAS(POS, FS)
    t0 = __import__("time").perf_counter()
    if name.startswith(("L0", "L1a", "L1b", "L1c", "L2")):
        if name == "L2_gridDAS":
            azs = np.arange(-180, 180, 10.0)
            pows = np.array([np.mean(das.process(mix, far_pos(a)) ** 2) for a in azs])
        elif name == "L1c_gridGSCgn":
            m = gain_normalize(mix)
            azs = np.arange(-180, 180, 10.0)
            gsc = GSC_MCWF(POS, FS)
            pows = np.array([np.mean(gsc.process(m, far_pos(a)) ** 2) for a in azs])
        else:
            kind = {"L0_gridLCMV": "lcmv", "L1a_gridPhaseOnly": "phaseonly",
                    "L1b_gridRobustLCMV": "robust"}[name]
            azs, pows = capon_map(mix, kind)
        sel = float(azs[np.argmax(pows)])
        y = extract_audio(name, mix, sel)
    elif name in ("L3a_music+GSCgn", "L3b_music+SubspaceNull+comb"):
        cands = music_candidates(mix, k=3)
        auds = {}
        for c in cands:
            y = extract_audio(name, mix, c)
            if name.endswith("comb"):
                f0, sc = track_f0(y)
                if f0 and sc > 3:
                    y = comb(y, f0)
            auds[c] = y
        sel = max(auds, key=lambda c: tonality8k(auds[c]))
        y = auds[sel]
    elif name == "L4_OverIVA":
        Z = stft(mix)
        Y = pra.bss.auxiva(np.transpose(Z, (2, 1, 0)), n_src=3, n_iter=50,
                           proj_back=True)
        ys = []
        for k in range(Y.shape[2]):
            ys.append(spsig.istft(Y[:, :, k].T, fs=FS, nperseg=NFFT,
                                  noverlap=NFFT - 256, window="hann")[1])
        k_sel = int(np.argmax([si_sdr(y, ref) if False else tonality8k(y) for y in ys]))
        y = ys[k_sel]
        from experiments.benchmark_unified import manifold_doa as _md
        _freqs = np.fft.rfftfreq(NFFT, 1 / FS)
        sel = _md(Z, y, _freqs, np.arange(-180, 180, 2.0))
    dt = __import__("time").perf_counter() - t0
    ya, ra = align(y, ref)
    return dict(Pipeline=name, SelAz=round(sel, 1), SelErr=ang(sel, AZ_T),
                SI_SDR=si_sdr(y, ref), Tonality=tonality8k(y), RTF=dt / DUR)


# ------------------------------------------------- gain-norm adversarial
def gain_norm_adversarial():
    print("\n" + "=" * 88)
    print("GAIN-NORM ADVERSARIAL: calibrate on a NEAR-FIELD-interferer scene,")
    print("apply to a FAR-FIELD scene (user's concern: acoustics baked into gains)")
    print("=" * 88)
    t = np.arange(int(FS * DUR)) / FS
    noise = 0.1 * rng.standard_normal((M, len(t)))
    # 'calibration' scene: quiet far target + LOUD near-field broadband at az 100
    cal = prop_far(gen_tonal(t) * 10 ** (-10 / 20), AZ_T) \
        + prop_near(gen_bb(t, seed=7), 100.0) * 10 ** (35 / 20) * 0.1 / np.sqrt(np.mean(gen_bb(t, seed=7) ** 2)) \
        + noise
    # evaluation scene: pure far-field target+interferer, TRUE gains = 1
    ev = prop_far(gen_tonal(t) * 10 ** (-10 / 20), AZ_T) \
        + prop_far(gen_bb(t, seed=8), -60) * 10 ** (5 / 20) * 0.1 / np.sqrt(np.mean(gen_bb(t, seed=8) ** 2)) \
        + noise
    ref = gen_tonal(t)

    gsc = GSC_MCWF(POS, FS)
    rms_near = np.sqrt(np.mean(cal ** 2, axis=1, keepdims=True))       # naive
    rms_far = np.sqrt(np.mean((prop_far(gen_tonal(t), 0) * 0 + noise) ** 2, axis=1, keepdims=True))
    # coherence-weighted: gain ratios vs mic 0 in coherent bands
    Zc = stft(cal)
    f_axis = np.fft.rfftfreq(NFFT, 1 / FS)
    band = (f_axis > 250) & (f_axis < 1100)
    P0 = np.mean(np.abs(Zc[0, band, :]) ** 2, axis=1)
    g_coh = np.ones((M, 1))
    for m in range(M):
        Pm = np.mean(np.abs(Zc[m, band, :]) ** 2, axis=1)
        S01 = np.abs(np.mean(Zc[m, band, :] * np.conj(Zc[0, band, :]), axis=1)) ** 2
        coh = S01 / (Pm * P0 + 1e-20)
        w = coh > 0.6
        if np.any(w):
            g_coh[m] = np.sqrt(np.mean(Pm[w]) / np.mean(P0[w]))

    rows = []
    for tag, g in [("no norm", np.ones((M, 1))),
                   ("RMS(near-field cal scene)", rms_near / np.sqrt(np.mean(rms_near ** 2))),
                   ("coherence-weighted", g_coh)]:
        y = gsc.process(ev * g, far_pos(AZ_T))
        rows.append((tag, si_sdr(y, ref)))
        print(f"  GSC on far-field eval, gains = {tag:32s}: SI-SDR = {rows[-1][1]:6.2f} dB")
    print("  (ideal = 'no norm' since true gains are 1; a bad estimator LOWERS it)")


# ------------------------------------------------- main
def main():
    scen_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08",
                "S09", "S10", "S11", "S12", "S13"]
    pipes = ["L0_gridLCMV", "L1a_gridPhaseOnly", "L1b_gridRobustLCMV",
             "L1c_gridGSCgn", "L2_gridDAS", "L3a_music+GSCgn",
             "L3b_music+SubspaceNull+comb", "L4_OverIVA"]
    rows = []
    for sid in scen_ids:
        mix, ref = build_scenario(sid)
        for p in pipes:
            try:
                r = run_pipeline(p, mix, ref)
                r["Scenario"] = sid
                rows.append(r)
                print(f"[{sid}] {p:28s} sel={r['SelAz']:6.1f} err={r['SelErr']:5.1f} "
                      f"SI-SDR={r['SI_SDR']:6.1f} tonal={r['Tonality']:5.1f}", flush=True)
            except Exception as e:
                print(f"[{sid}] {p} FAILED {type(e).__name__}: {e}", flush=True)
                rows.append(dict(Scenario=sid, Pipeline=p, SelAz=np.nan, SelErr=np.nan,
                                 SI_SDR=np.nan, Tonality=np.nan, RTF=np.nan))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "all_pipelines_benchmark.csv"), index=False)

    print("\n" + "=" * 96)
    print("FULL MATRIX (SelErr<15 = target chosen; SI-SDR of delivered audio; tonality)")
    print("=" * 96)
    piv = df.pivot_table(index="Scenario", columns="Pipeline",
                         values=["SelErr", "SI_SDR"])
    print(piv.to_string(float_format=lambda v: f"{v:7.1f}"))

    gain_norm_adversarial()


if __name__ == "__main__":
    main()

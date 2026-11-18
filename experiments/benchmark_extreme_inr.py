"""
Extreme-INR benchmark: close loud interferer vs distant weak target.

Regime: dog-bark-like bursts at 10 m (near-field, spherical) vs siren-like
harmonic target at 2 km (far-field). Interferer +40/+50 dB over target,
target 20 dB BELOW sensor noise at the mics. Real 64-mic geometry, 8 kHz,
alias-safe band.

Questions answered:
 1. How deep can spatial nulling reject the bark? (DAS vs LCMV vs LCMV+gn
    vs GSC+gn vs rank-k subspace null, all steered at the target)
 2. What does a harmonic-comb post-filter add on top of the beamformer?
 3. What survives in the worst case (interferer in the SAME beam)?

Purity metric: LS decomposition of every output onto the aligned clean
target and bark references -> target retention / bark residual / dSIR.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as spsig

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_pipelines import purity

FS = 8000
DUR = 3.0
C = 343.0
NFFT = 512
HOP = 256

geo = np.loadtxt(os.path.join(REPO, "array_geo.csv"), delimiter=",")
POS = geo.T  # (64, 3)
M = POS.shape[0]

AZ_T = 40.0            # target azimuth (2 km, far field)
F0S = [350.0, 700.0, 1050.0]  # siren harmonics (alias-safe band)
rng = np.random.default_rng(3)


def far_pos(az, dist=2000.0):
    a = np.deg2rad(az)
    return dist * np.array([np.cos(a), np.sin(a), 0.0])


def frac_delay(sig, tau):
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1 / FS)
    return np.fft.irfft(S * np.exp(-2j * np.pi * f * tau), n=len(sig))


def steering_far(az):
    u = far_pos(az)
    u /= np.linalg.norm(u)
    tau = -(POS @ u) / C
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    f[0] = 1e-6
    return np.exp(-1j * 2 * np.pi * np.outer(f, tau)) / np.sqrt(M)


def stft(x):
    return spsig.stft(x, fs=FS, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")[2]


def istft(Z):
    return spsig.istft(Z, fs=FS, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")[1]


def make_scene(int_az, inr_db):
    """Target: siren harmonics at AZ_T, 2 km. Interferer: bark bursts,
    near-field 10 m at int_az. Target is 20 dB below sensor noise at mics."""
    t = np.arange(int(FS * DUR)) / FS
    env = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 0.9 * t + 0.4) * 0.8, -1, 1)
    siren = sum(np.sin(2 * np.pi * f0 * t) for f0 in F0S) / len(F0S) * env
    siren *= 10 ** (-20 / 20)  # target 20 dB below noise at mic scale

    bark_src = np.zeros(len(t))
    b, a = spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)], btype="band")
    for start in (0.3, 0.9, 1.1, 1.7, 2.3, 2.5):
        n = int(0.12 * FS)
        i0 = int(start * FS)
        burst = rng.standard_normal(min(n, len(t) - i0))
        bark_src[i0:i0 + len(burst)] += burst
    bark_src = spsig.filtfilt(b, a, bark_src)

    # far-field target (plane-wave delays), near-field interferer (spherical)
    p_far = far_pos(AZ_T)
    x_t = np.stack([frac_delay(siren, -(POS[m] @ (p_far / 2000.0)) / C)
                    for m in range(M)])
    p_near = far_pos(int_az, dist=10.0)
    d = np.linalg.norm(POS - p_near, axis=1)
    x_i = np.stack([frac_delay(bark_src, d[m] / C) / (4 * np.pi * d[m])
                    for m in range(M)])
    x_i *= np.sqrt(np.mean(d ** 2))  # undo mean 1/r so gain is explicit

    x_i *= 10 ** (inr_db / 20) * 0.1 / np.sqrt(np.mean(x_i ** 2))  # rel noise floor
    noise = 0.1 * rng.standard_normal((M, len(t)))
    return x_t + x_i + noise, siren, bark_src


def gain_norm(x):
    r = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))
    return x / (r + 1e-12)


def beamform(cls_or_fn, mix, look_az):
    return cls_or_fn(mix, look_az)


def run_repo(bf, mix, look_az):
    return bf.process(mix, far_pos(look_az))


def subspace_null(mix, look_az, k=2):
    """Data-driven rank-k interference-subspace projection steered at target."""
    Z = stft(mix)  # (M, F, T)
    A = steering_far(look_az)  # (F, M)
    Y = np.zeros_like(Z[0])
    _, F, T = Z.shape
    for f in range(F):
        X = Z[:, f, :]
        R = (X @ X.conj().T) / T
        evals, evecs = np.linalg.eigh(R)
        V = evecs[:, -k:]                      # dominant (interferer) subspace
        Pv = V @ V.conj().T
        a = A[f]
        a_perp = a - Pv @ a
        denom = np.real(a.conj() @ a_perp)
        if denom < 1e-12:
            w = a / M
        else:
            w = a_perp / denom
        Y[f] = w.conj() @ X
    return istft(Y)


def comb_postfilter(y, f0s=F0S, floor=0.05, width_hz=25.0):
    """Harmonic-comb Wiener-style mask on the beamformed output."""
    Z = stft(y[np.newaxis, :])[0]  # (F, T)
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    mask = np.full((len(f), Z.shape[1]), floor)
    for f0 in f0s:
        for h in range(1, 4):
            fh = f0 * h
            if fh > f[-1]:
                break
            mask[np.abs(f - fh) < width_hz] = 1.0
    return istft(Z * mask)


def tonal_metrics(y, f0s=F0S, bw=12.0):
    """Robust detectability metric for this regime: power at the target's
    harmonic lines vs control bands (noise + bark residual). Immune to the
    correlation-alignment failure that breaks LS purity when the error
    power dwarfs the target."""
    f, P = spsig.periodogram(np.asarray(y).flatten(), FS, nfft=32768,
                             window="blackman")
    lines, ctrl = [], []
    for f0 in f0s:
        for h in (1, 2, 3):
            fh = f0 * h
            if fh < FS / 2 - 50:
                lines.append(fh)
                ctrl.append(fh * 1.18)  # off-line control frequency
    mask_l = np.zeros_like(f, bool)
    mask_c = np.zeros_like(f, bool)
    for fc in lines:
        mask_l |= np.abs(f - fc) < bw
    for fc in ctrl:
        mask_c |= np.abs(f - fc) < bw
    p_sig = float(np.sum(P[mask_l]))
    p_res = float(np.sum(P[mask_c]))
    return 10 * np.log10(p_sig / (p_res + 1e-20))


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def evaluate(y, refs, name, case, inr, rows, ref_tonal):
    t_snr = tonal_metrics(y)
    rows.append(dict(Case=case, INR=inr, Method=name,
                     OutSNR_tonal=t_snr,
                     Improvement=t_snr - ref_tonal))


def main():
    rows = []
    for int_az, case in [(-60.0, "separated(100deg)"), (AZ_T, "aligned(same beam)")]:
        for inr in (40, 50):
            mix, siren, bark = make_scene(int_az, inr)
            refs = (siren, bark)
            ref_tonal = tonal_metrics(mix[0])
            print(f"[scene] {case} INR=+{inr} dB  (mic-0 tonal SNR {ref_tonal:.1f} dB)", flush=True)

            das = StandardDAS(POS, FS)
            lcmv = LCMVSMI(POS, FS)
            gsc = GSC_MCWF(POS, FS)
            mix_gn = gain_norm(mix)

            cands = [
                ("DAS", run_repo(das, mix, AZ_T)),
                ("LCMV", run_repo(lcmv, mix, AZ_T)),
                ("LCMV+gn", run_repo(lcmv, mix_gn, AZ_T)),
                ("GSC+gn", run_repo(gsc, mix_gn, AZ_T)),
                ("SubspaceNull(k=2)", subspace_null(mix, AZ_T, k=2)),
                ("SubspaceNull(k=3)", subspace_null(mix, AZ_T, k=3)),
            ]
            for name, y in cands:
                evaluate(y, refs, name, case, inr, rows, ref_tonal)
                evaluate(comb_postfilter(y), refs, name + "+comb", case, inr, rows, ref_tonal)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "extreme_inr_benchmark.csv"), index=False)

    print("\n" + "=" * 96)
    print("EXTREME-INR: tonal SNR at output (dB) and improvement over raw mic (dB)")
    print("=" * 96)
    for (case, inr), sub in df.groupby(["Case", "INR"]):
        print(f"\n--- {case} @ interferer +{inr} dB ---")
        for _, r in sub.sort_values("Improvement", ascending=False).iterrows():
            print(f"  {r.Method:24s} outSNR={r.OutSNR_tonal:7.1f} dB   "
                  f"improvement={r.Improvement:7.1f} dB")


if __name__ == "__main__":
    main()

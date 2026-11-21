"""
Phase-miscalibration robustness + LCMV waveform-fidelity benchmark.

The system has NO calibration. Tested error models (applied per mic):
  - delay/position error: random +/- d cm (phase grows with frequency)
  - all-pass phase error: random +/- phi degrees (constant over frequency)

Per error level x method (DAS, LCMV, RobustLCMV, GSC, PhaseOnly, SubspaceNull):
  - DOA hit (NormMUSIC finds the target within 10 deg)
  - SI-SDR of extraction at the true azimuth
  - LSD (log-spectral distance vs the clean target): waveform fidelity --
    LCMV-type filters apply different complex gains per frequency bin; with
    steering mismatch the distortionless constraint no longer holds at the
    TRUE steering, so the target spectrum gets colored.
  - line-gain spread: gain deviation across the target's harmonic lines
    (direct measure of per-bin coloration; DAS is flat by construction)
  - tonality (harmonic contrast; proxy for LAM/AST response quality --
    the real transformers/AST stack is not installed in this environment)
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
from simulator.beamformers.robust_lcmv import RobustLCMV
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from experiments.benchmark_extreme_inr import (
    POS, FS, DUR, C, NFFT, far_pos, frac_delay, subspace_null,
)

M = POS.shape[0]
AZ_T = 40.0
AZ_I = -60.0
F0S = [350.0, 700.0, 1050.0]
N_DRAWS = 3

DELAY_ERRORS = [(0.0, "0cm"), (0.5, "0.5cm"), (1.0, "1cm"), (2.0, "2cm"), (4.0, "4cm")]
PHASE_ERRORS = [(10.0, "10deg"), (20.0, "20deg"), (45.0, "45deg")]


def make_scene(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * DUR)) / FS
    r = np.random.default_rng(seed + 1)
    env = 0.5 + 0.5 * np.clip(0.8 * np.sin(2 * np.pi * 0.9 * t + r.uniform(0, 6)), -1, 1)
    target = sum(np.sin(2 * np.pi * f0 * t) for f0 in F0S) / 3.0 * env  # +10 dB over noise
    target *= 10 ** (10 / 20) * 0.1
    interf = spsig.filtfilt(*spsig.butter(4, [300 / (FS / 2), 3000 / (FS / 2)],
                                          btype="band"),
                            rng.standard_normal(len(t)))
    interf *= 10 ** (15 / 20) * 0.1 / np.sqrt(np.mean(interf ** 2))

    def prop(sig, az):
        u = far_pos(az) / 2000.0
        return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])

    return prop(target, AZ_T) + prop(interf, AZ_I) + 0.1 * rng.standard_normal((M, len(t))), target


def apply_delay_err(mix, rng, d_cm):
    out = np.zeros_like(mix)
    for m in range(M):
        out[m] = frac_delay(mix[m], rng.uniform(-d_cm, d_cm) / 100.0 / C)
    return out


def apply_phase_err(mix, rng, phi_deg):
    Z = spsig.stft(mix, fs=FS, nperseg=NFFT, noverlap=NFFT - 256, window="hann")[2]
    ph = np.exp(1j * np.deg2rad(rng.uniform(-phi_deg, phi_deg, size=(M, 1, 1))))
    return spsig.istft(Z * ph, fs=FS, nperseg=NFFT, noverlap=NFFT - 256,
                       window="hann")[1][:, :mix.shape[1]]


def align(y, ref):
    n = min(len(y), len(ref))
    c = spsig.correlate(y[:n] - y[:n].mean(), ref[:n] - ref[:n].mean(),
                        mode="full", method="fft")
    lag = int(np.argmax(np.abs(c)) - (n - 1))
    if lag > 0:
        y, ref = y[lag:], ref[:len(y) - lag]
    elif lag < 0:
        y, ref = y[:len(y) + lag], ref[-lag:]
    m = min(len(y), len(ref))
    return y[:m], ref[:m]


def si_sdr(y, ref):
    ya, ra = align(np.asarray(y).flatten(), ref)
    a = np.dot(ya, ra) / (np.dot(ya, ya) + 1e-12)
    e = a * ya - ra
    return 10 * np.log10(np.dot(ra, ra) / (np.dot(e, e) + 1e-12))


def lsd(y, ref):
    ya, ra = align(np.asarray(y).flatten(), ref)
    nper = 2048
    f, t, Sy = spsig.stft(ya, fs=FS, nperseg=nper, window="hann")
    _, _, Sr = spsig.stft(ra, fs=FS, nperseg=nper, window="hann")
    band = (f > 200) & (f < 3000)
    d = np.abs(10 * np.log10(np.abs(Sy[band]) ** 2 + 1e-12)
               - 10 * np.log10(np.abs(Sr[band]) ** 2 + 1e-12))
    return float(np.mean(d))


def line_gain_spread(y, ref):
    ya, ra = align(np.asarray(y).flatten(), ref)
    f, Py = spsig.periodogram(ya, FS, nfft=32768, window="blackman")
    _, Pr = spsig.periodogram(ra, FS, nfft=32768, window="blackman")
    gains = []
    for f0 in F0S:
        sel = np.abs(f - f0) < 10
        gains.append(10 * np.log10(np.sum(Py[sel]) / (np.sum(Pr[sel]) + 1e-20)))
    return float(max(gains) - min(gains))


def tonality(y, f0s=F0S):
    f, P = spsig.periodogram(np.asarray(y).flatten(), FS, nfft=32768,
                             window="blackman")
    lines = []
    ctrl = []
    for f0 in f0s:
        for h in (1, 2, 3):
            if f0 * h < FS / 2 - 50:
                lines.append(f0 * h)
                ctrl.append(f0 * h * 1.18)
    ml = np.zeros_like(f, bool)
    mc = np.zeros_like(f, bool)
    for fc in lines:
        ml |= np.abs(f - fc) < 12
    for fc in ctrl:
        mc |= np.abs(f - fc) < 12
    return float(10 * np.log10(np.sum(P[ml]) / (np.sum(P[mc]) + 1e-20)))


def doa_hit(mix):
    import pyroomacoustics as pra
    Z = spsig.stft(mix, fs=FS, nperseg=NFFT, noverlap=NFFT - 256, window="hann")[2]
    d = pra.doa.NormMUSIC(POS.T.copy(), FS, nfft=NFFT,
                          azimuth=np.deg2rad(np.arange(-180, 180, 2.0)))
    d.locate_sources(Z, num_src=2, freq_range=[250.0, 1100.0])
    est = np.rad2deg(np.atleast_1d(d.azimuth_recon))

    def ang(a, b):
        dd = abs(a - b) % 360.0
        return min(dd, 360.0 - dd)
    return int(any(ang(e, AZ_T) < 10 for e in est))


def main():
    rows = []
    methods = [("DAS", StandardDAS(POS, FS)), ("LCMV", LCMVSMI(POS, FS)),
               ("RobustLCMV", RobustLCMV(POS, FS)), ("GSC", GSC_MCWF(POS, FS)),
               ("PhaseOnly", PhaseOnlyLCMV(POS, FS))]
    for kind, errs in (("delay_cm", DELAY_ERRORS), ("phase_deg", PHASE_ERRORS)):
        for lvl, lbl in errs:
            for draw in range(N_DRAWS):
                seed = 4000 + draw
                mix, ref = make_scene(seed)
                rng = np.random.default_rng(seed + 50)
                if lvl > 0:
                    mix = (apply_delay_err(mix, rng, lvl) if kind == "delay_cm"
                           else apply_phase_err(mix, rng, lvl))
                hit = doa_hit(mix)
                for name, bf in methods:
                    y = bf.process(mix, far_pos(AZ_T))
                    rows.append(dict(Error=kind, Level=lbl, Draw=draw, Method=name,
                                     DOA=hit, SI_SDR=si_sdr(y, ref), LSD=lsd(y, ref),
                                     LineSpread=line_gain_spread(y, ref),
                                     Tonality=tonality(y)))
                y = subspace_null(mix, AZ_T, k=2)
                rows.append(dict(Error=kind, Level=lbl, Draw=draw, Method="SubspaceNull",
                                 DOA=hit, SI_SDR=si_sdr(y, ref), LSD=lsd(y, ref),
                                 LineSpread=line_gain_spread(y, ref),
                                 Tonality=tonality(y)))
                print(f"[{kind}={lbl}] draw {draw}: DOA={hit} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "docs", "phase_robustness.csv"), index=False)
    print("\n" + "=" * 100)
    print("PHASE-MISCALIBRATION ROBUSTNESS (means over 3 draws)")
    print("=" * 100)
    g = df.groupby(["Error", "Level", "Method"]).agg(
        {"SI_SDR": "mean", "LSD": "mean", "LineSpread": "mean", "Tonality": "mean"})
    print(g.to_string(float_format=lambda v: f"{v:7.2f}"))


if __name__ == "__main__":
    main()

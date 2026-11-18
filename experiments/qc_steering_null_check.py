"""
QC experiment: steering sign + null depth (SSR_Depth) investigation.

1. Synthesize a physically correct 8-mic scene (FFT fractional delays):
   target at az=+40, interferer at az=-60, both el=0.
2. Sweep look azimuth with the repo's own beamformers:
   - where does DAS actually peak? (mirroring test)
   - SSR_Depth (max/min output power over grid) for LCMV-SMI vs PhaseOnly,
     original vs sign-fixed steering, with/without random per-mic gains.
3. Cross-check with off-the-shelf pyroomacoustics DOA (SRP / MUSIC).
"""
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import scipy.signal as sig
from simulator.beamformers.base import BaseBeamformer
from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_phase import PhaseOnlyLCMV

rng = np.random.default_rng(0)
fs = 16000
DUR = 3.0
M = 8
R_ARRAY = 0.08  # m -> spacing 6.1 cm -> spatial alias above ~2.8 kHz
C = 343.0

ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
POS = np.stack([R_ARRAY * np.cos(ang), R_ARRAY * np.sin(ang), np.zeros(M)], axis=1)


def src_pos(az_deg, el_deg=0.0, dist=120.0):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def frac_delay(sig, tau_s, fs):
    """Physical propagation: x(t) = s(t - tau)."""
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    return np.fft.irfft(S * np.exp(-2j * np.pi * f * tau_s), n=len(sig))


def make_scene(gain_mismatch=False):
    t = np.arange(int(fs * DUR)) / fs
    target = sum(np.sin(2 * np.pi * f * t) for f in [400, 800, 1200, 1600]) / 4.0
    interferer = sum(np.sin(2 * np.pi * f * t + 0.7) for f in [550, 1100, 1650]) / 4.0

    def propagate(sig, az):
        pos_s = src_pos(az)
        cols = []
        for m in range(M):
            d = np.linalg.norm(POS[m] - pos_s)
            cols.append(frac_delay(sig, d / C, fs))
        return np.stack(cols)

    x_t = propagate(target, 40.0)
    x_i = propagate(interferer, -60.0)
    x_t *= np.sqrt(2.0) / np.sqrt(np.mean(x_t**2))   # interferer 3 dB louder
    x_i *= 2.0 * np.sqrt(2.0) / np.sqrt(np.mean(x_i**2)) / np.sqrt(2.0)
    noise = 0.02 * rng.standard_normal((M, len(t)))
    audio = x_t + x_i + noise
    if gain_mismatch:
        g = rng.uniform(0.25, 1.0, size=(M, 1))
        audio = audio * g
    return audio, x_t, x_i


def az_dirs():
    return np.arange(-180, 180, 2.5)


def class_power_sweep(cls, audio):
    bf = cls(POS, fs)
    p = []
    for az in az_dirs():
        y = bf.process(audio, src_pos(az))
        p.append(float(np.mean(y**2)))
    return np.array(p)


# ---- vectorized (formula-faithful) LCMV & PhaseOnly -------------------------
def precompute(audio):
    _, _, Z = sig.stft(audio, fs=fs, nperseg=512, noverlap=256, window="hann")
    return Z  # (M, F, T)


def steering(az_deg, conj_fix=False):
    u = src_pos(az_deg)
    u /= np.linalg.norm(u)
    delays = -(POS @ u) / C                      # channel.py far-field convention
    f = np.fft.rfftfreq(512, 1 / fs)
    f[0] = 1e-6
    ph = np.exp(1j * 2 * np.pi * np.outer(f, delays))  # base.py: a = e^{+j w tau}
    a = ph / np.sqrt(M)
    return np.conj(a) if conj_fix else a


def sweep_lcmv(Z, conj_fix=False, eps=1e-3):
    Xs = Z  # (M,F,T)
    Mm, F, T = Z.shape
    a_all = {az: steering(az, conj_fix) for az in az_dirs()}
    out = {}
    Rinv = np.zeros((F, Mm, Mm), complex)
    for f in range(F):
        X = Xs[:, f, :]
        R = (X @ X.conj().T) / T + eps * np.trace(X @ X.conj().T) / T * np.eye(Mm)
        Rinv[f] = np.linalg.inv(R)
    for az, a in a_all.items():
        e = 0.0
        for f in range(F):
            w = Rinv[f] @ a[f]
            w = w / (a[f].conj() @ w + 1e-30)
            e += float(np.sum(np.abs(w.conj() @ Xs[:, f, :]) ** 2))
        out[az] = e
    return out


def sweep_phaseonly(Z, conj_fix=False):
    """Replicates robust_phase.py exactly, incl. Ledoit-Wolf shrinkage (vectorized var est)."""
    Mm, F, T = Z.shape
    mags = np.abs(Z) + 1e-12
    Zp = Z / mags
    a_all = {az: steering(az, conj_fix) for az in az_dirs()}
    shrink = []
    Rinv = np.zeros((F, Mm, Mm), complex)
    for f in range(F):
        Xp = Zp[:, f, :]
        R = (Xp @ Xp.conj().T) / T
        mu = np.trace(R).real / Mm
        F_norm_sq = np.sum(np.abs(R - mu * np.eye(Mm)) ** 2)
        e_t = np.sum(np.abs(Xp) ** 2, axis=0)          # ||x_t||^2
        var_est = (np.sum(e_t**2) - T * np.sum(np.abs(R) ** 2)) / T**2
        s = float(np.clip(var_est.real / max(F_norm_sq, 1e-30), 0, 1)) if F_norm_sq > 0 else 1.0
        shrink.append(s)
        Rload = (1 - s) * R + s * mu * np.eye(Mm)
        Rinv[f] = np.linalg.inv(Rload)
    out = {}
    for az, a in a_all.items():
        e = 0.0
        for f in range(F):
            w = Rinv[f] @ a[f]
            w = w / (a[f].conj() @ w + 1e-30)
            e += float(np.sum(np.abs(w.conj() @ Z[:, f, :]) ** 2))   # applied to ORIGINAL Z
        out[az] = e
    return out, np.array(shrink)


def report(name, powers):
    azs = az_dirs()
    v = np.array([powers[a] for a in azs])
    peak = azs[np.argmax(v)]
    ssr = 10 * np.log10(np.max(v) / max(np.min(v), 1e-30))
    null = 10 * np.log10(v[np.argmin(np.abs(azs - (-60)))] / v[np.argmin(np.abs(azs - 40))])
    print(f"  {name:34s} peak@az={peak:7.1f}  SSR_depth={ssr:7.2f} dB  P(-60)/P(40)={null:7.2f} dB")
    return v


def main():
    print(f"numpy {np.__version__}; scene: target az=+40 (harmonics), interferer az=-60 (3 dB louder), el=0, M={M}, r={R_ARRAY} m")

    audio, x_t, x_i = make_scene(gain_mismatch=False)

    print("\n[A] DAS sweep with repo code (original steering) vs sign-fixed:")
    p = class_power_sweep(StandardDAS, audio)
    azs = az_dirs()
    print(f"  DAS (as-is)      peak@az={azs[np.argmax(p)]:7.1f} deg   (true target az = +40)")
    orig = BaseBeamformer.compute_steering_vector
    BaseBeamformer.compute_steering_vector = lambda self, lp, is_spherical=False: np.conj(orig(self, lp, is_spherical))
    p2 = class_power_sweep(StandardDAS, audio)
    print(f"  DAS (conj-fixed) peak@az={azs[np.argmax(p2)]:7.1f} deg")
    BaseBeamformer.compute_steering_vector = orig

    print("\n[B] Null depth / SSR over az grid (formula-faithful replication):")
    Z = precompute(audio)
    report("LCMV-SMI (as-is steering)", sweep_lcmv(Z))
    report("LCMV-SMI (sign-fixed)", sweep_lcmv(Z, conj_fix=True))
    po, shr = sweep_phaseonly(Z)
    report("PhaseOnlyLCMV (as-is)", po)
    print(f"     -> Ledoit-Wolf shrinkage: mean={shr.mean():.3f}  max={shr.max():.3f}  (1.0 == pure DAS, no nulls)")
    report("PhaseOnlyLCMV (sign-fixed)", sweep_phaseonly(Z, conj_fix=True)[0])

    # consistency check: vectorized LCMV pattern must match the actual class
    cls_p = class_power_sweep(LCMVSMI, audio)
    azs = az_dirs()
    probe_az = (-60, -20, 0, 40, 120)
    idx = [int(np.argmin(np.abs(azs - a))) for a in probe_az]
    mine = sweep_lcmv(Z)
    cls_v = cls_p[idx]
    my_v = np.array([mine[azs[i]] for i in idx])
    scale = cls_v[3] / my_v[3]  # fit one scale factor at the target direction
    rel = np.abs(cls_v / (scale * my_v) - 1.0)
    print(f"\n[C] Faithfulness check (LCMV class vs vectorized after single-scale fit):")
    for a, rv, r in zip(probe_az, my_v, rel):
        print(f"     az={a:6.1f}: class={cls_v[list(probe_az).index(a)]:.3e}  mine={rv*scale:.3e}  rel.err={r:.1e}")

    print("\n[D] Same with injected random per-mic gains U(0.25,1.0) (real-world eval stress):")
    audio2, _, _ = make_scene(gain_mismatch=True)
    Z2 = precompute(audio2)
    report("LCMV-SMI (sign-fixed)", sweep_lcmv(Z2, conj_fix=True))
    report("PhaseOnlyLCMV (sign-fixed)", sweep_phaseonly(Z2, conj_fix=True)[0])
    report("LCMV-SMI (as-is steering)", sweep_lcmv(Z2))

    print("\n[E] Off-the-shelf pyroomacoustics on the same scene (clean):")
    try:
        import pyroomacoustics as pra
        print("  pra.doa:", [m for m in dir(pra.doa) if m[0].isupper()])
        print("  pra.beamforming Beamformer methods:",
              [m for m in dir(pra.beamforming.Beamformer) if not m.startswith("_")])
        try:
            print("  pra.bss:", [m for m in dir(pra.bss) if not m.startswith("_")])
        except AttributeError:
            pass
        mics = pra.MicrophoneArray(POS.T.copy(), fs)
        z = np.transpose(Z, (1, 0, 2))  # (F, M, T)
        az_grid = np.deg2rad(np.arange(-180, 180, 2.0))
        for cls_name in ("SRP", "NormMUSIC", "FRIDA"):
            try:
                cls = getattr(pra.doa, cls_name)
                kw = {"nfft": 512, "mode": "far"} if cls_name == "SRP" else {"nfft": 512}
                try:
                    d = cls(mics, fs, azimuth=az_grid, **kw)
                except TypeError:
                    d = cls(mics, fs, **kw)
                d.locate_sources(z=z, num_src=2)
                rec = np.rad2deg(np.atleast_1d(d.azimuth_recon))
                print(f"  {cls_name}.locate_sources -> azimuth(s) deg: {np.round(rec, 1)}  (truth: 40 and -60)")
            except Exception as e:
                print(f"  {cls_name} failed: {type(e).__name__}: {e}")
    except ImportError as e:
        print("  pyroomacoustics not importable:", e)


if __name__ == "__main__":
    main()

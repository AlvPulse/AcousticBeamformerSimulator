"""
Gen-2 steering regression tests.

Synthesizes scenes with physically correct FFT fractional delays (independent
of simulator.acoustics.scene_generator) and verifies that gen-2 beamformers
steer at the TRUE direction -- not the mirrored one.

These tests would have caught the conjugation bug in
BaseBeamformer.compute_steering_vector (a = exp(+j w tau) instead of
exp(-j w tau)), which mirrored every geometry-trusting beamformer 180 deg.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI

FS = 16000
M = 8
R_ARRAY = 0.08  # m; keeps spatial aliasing above ~2.8 kHz
C = 343.0

_ANG = np.linspace(0, 2 * np.pi, M, endpoint=False)
POS = np.stack([R_ARRAY * np.cos(_ANG), R_ARRAY * np.sin(_ANG), np.zeros(M)], axis=1)


def _src_pos(az_deg, el_deg=0.0, dist=120.0):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _frac_delay(sig, tau_s):
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1.0 / FS)
    return np.fft.irfft(S * np.exp(-2j * np.pi * f * tau_s), n=len(sig))


def _scene(target_az, interferer_az, rng, interferer_gain_db=3.0):
    t = np.arange(int(FS * 2.0)) / FS
    target = sum(np.sin(2 * np.pi * f * t) for f in [400, 800, 1200, 1600]) / 4.0
    interferer = sum(np.sin(2 * np.pi * f * t + 0.7) for f in [550, 1100, 1650]) / 4.0

    def propagate(sig, az):
        p = _src_pos(az)
        cols = [_frac_delay(sig, np.linalg.norm(POS[m] - p) / C) for m in range(M)]
        x = np.stack(cols)
        return x / np.sqrt(np.mean(x**2))

    x_t = propagate(target, target_az)
    x_i = propagate(interferer, interferer_az) * 10 ** (interferer_gain_db / 20)
    noise = 0.02 * rng.standard_normal((M, len(t)))
    return x_t + x_i + noise, target


def _das_peak_az(bf, audio, el=0.0):
    azs = np.arange(-180, 180, 2.5)
    powers = [float(np.mean(bf.process(audio, _src_pos(a, el)) ** 2)) for a in azs]
    return azs[int(np.argmax(powers))]


def test_das_peaks_at_true_doa():
    rng = np.random.default_rng(1)
    audio, _ = _scene(40.0, -60.0, rng, interferer_gain_db=-6.0)  # target dominant
    peak = _das_peak_az(StandardDAS(POS, FS), audio)
    assert abs(peak - 40.0) <= 2.5, f"DAS peaked at {peak} deg, expected ~40 (mirroring bug?)"


def test_das_not_mirrored():
    rng = np.random.default_rng(2)
    audio, _ = _scene(40.0, -60.0, rng, interferer_gain_db=-6.0)
    bf = StandardDAS(POS, FS)
    p_true = float(np.mean(bf.process(audio, _src_pos(40.0)) ** 2))
    p_mirror = float(np.mean(bf.process(audio, _src_pos(220.0)) ** 2))
    # A planar array at el=0 keeps a J0(2kr) mirror response, so the margin is
    # modest (~6-7 dB here) -- but ~0 dB would indicate the conjugation bug.
    assert 10 * np.log10(p_true / p_mirror) > 5.0, (
        f"Response at true DOA only {10*np.log10(p_true/p_mirror):.1f} dB above mirror"
    )


def test_lcmv_nulls_interferer():
    """Interferer-only scene: LCMV steered elsewhere must null it deep below DAS."""
    rng = np.random.default_rng(3)
    audio, _ = _scene(40.0, -60.0, rng, interferer_gain_db=6.0)
    interferer_only = audio.copy()
    # remove the target by resynthesizing without it
    t = np.arange(int(FS * 2.0)) / FS
    interferer = sum(np.sin(2 * np.pi * f * t + 0.7) for f in [550, 1100, 1650]) / 4.0
    p = _src_pos(-60.0)
    cols = [_frac_delay(interferer, np.linalg.norm(POS[m] - p) / C) for m in range(M)]
    x_i = np.stack(cols)
    x_i = x_i / np.sqrt(np.mean(x_i**2)) * 10 ** (6.0 / 20)
    interferer_only = x_i + 0.02 * rng.standard_normal((M, len(t)))

    das = StandardDAS(POS, FS)
    lcmv = LCMVSMI(POS, FS)
    y_das = das.process(interferer_only, _src_pos(40.0))
    y_lcmv = lcmv.process(interferer_only, _src_pos(40.0))
    null_db = 10 * np.log10(np.mean(y_lcmv**2) / np.mean(y_das**2))
    assert null_db < -15.0, f"LCMV null on interferer only {null_db:.1f} dB below DAS"


def test_lcmv_preserves_target():
    """Target-only scene: LCMV steered at the target must pass it (no self-cancellation)."""
    rng = np.random.default_rng(4)
    audio, _ = _scene(40.0, -60.0, rng, interferer_gain_db=-120.0)  # interferer off
    das = StandardDAS(POS, FS)
    lcmv = LCMVSMI(POS, FS)
    y_das = das.process(audio, _src_pos(40.0))
    y_lcmv = lcmv.process(audio, _src_pos(40.0))
    ratio_db = 10 * np.log10(np.mean(y_lcmv**2) / np.mean(y_das**2))
    assert ratio_db > -3.0, f"LCMV target response {ratio_db:.1f} dB below DAS (self-cancellation)"

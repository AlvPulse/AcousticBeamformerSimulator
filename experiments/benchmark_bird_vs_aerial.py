"""
Bird-vs-aerial detection benchmark (the actual mission): special birds
vs UAVs / airplanes / helicopters / engines.

Models: AST-10-10 (HF, cached) + YAMNet-ONNX (16 MB, independent
architecture). Both score the SAME audio; the detectability metric is
the bird-vs-aerial MARGIN: P(bird classes) - max P(engine/aerial/vehicle
classes) per model. A method "preserves detectability" if the extracted
audio keeps a positive margin (the classifier still sees a bird, not an
engine).

Scenes: REAL ESC-50 bird clip as target (real SI-SDR reference!), real
helicopter (near-field +40 dB) and engine (+30 dB) as interference; plus
a SYNTHETIC QUADROTOR (rotor-blade harmonic comb) variant, since UAVs are
tonal-comb sources distinct from broadband engines.
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
import librosa

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_extreme_inr import (
    POS, FS, DUR, C, far_pos, frac_delay, steering_far, subspace_null,
)
from experiments.benchmark_all_pipelines import gain_normalize, tonality8k
from experiments.benchmark_phase_robustness import align, si_sdr
from experiments.benchmark_pipelines import purity

M = POS.shape[0]
AZ_T, AZ_I = 40.0, -60.0
BIRD_CLASSES = ("bird", "wild animal", "animal", "chirp", "tweet")
AERIAL_CLASSES = ("engine", "vehicle", "helicopter", "aircraft", "fixed-wing",
                  "truck", "idling", "motor vehicle")


def load_model_map():
    from transformers import pipeline
    ast = pipeline("audio-classification",
                   model="MIT/ast-finetuned-audioset-10-10-0.4593")
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(REPO, "data", "models", "yamnet_onnx",
                                            "yamnet.onnx"),
                                providers=["CPUExecutionProvider"])
    yam_names = [row[2] for row in list(csv_reader(os.path.join(
        REPO, "data", "models", "yamnet_onnx", "yamnet_class_map.csv")))[1:]]
    return ast, sess, yam_names


def csv_reader(p):
    import csv
    return list(csv.reader(open(p)))


def score_ast(ast, y):
    y16 = librosa.resample(np.asarray(y).flatten(), orig_sr=FS, target_sr=16000)
    y16 = y16 / (np.max(np.abs(y16)) + 1e-9) * 0.9
    res = ast({"raw": y16, "sampling_rate": 16000}, top_k=8)
    return [(r["label"], float(r["score"])) for r in res]


def score_yamnet(sess, yam_names, y):
    w = librosa.resample(np.asarray(y, dtype="float32").flatten(),
                         orig_sr=FS, target_sr=16000)
    w = w / (np.max(np.abs(w)) + 1e-9)
    s = sess.run(None, {"waveform": w})[0]
    m = s.mean(axis=0)
    return [(yam_names[i], float(m[i])) for i in np.argsort(m)[::-1][:8]]


def margin(scores):
    pb = sum(s for lab, s in scores if any(b in lab.lower() for b in BIRD_CLASSES))
    pa = max((s for lab, s in scores
              if any(a in lab.lower() for a in AERIAL_CLASSES)), default=0.0)
    return pb - pa


def esc_clip(category, idx=0):
    meta = pd.read_csv(os.path.join(REPO, "ESC-50-master", "meta", "esc50.csv"))
    rows = meta[(meta.category == category) & (meta.fold != 0)].reset_index()
    fn = rows.iloc[idx % len(rows)].filename
    y, _ = librosa.load(os.path.join(REPO, "ESC-50-master", "audio", fn),
                        sr=FS, mono=True)
    # trim leading/trailing silence-tolerant: use the loudest 3 s window
    n = int(DUR * FS)
    if len(y) <= n:
        return np.pad(y, (0, n - len(y))), fn
    win = np.convolve(y ** 2, np.ones(FS), "valid")
    i0 = int(np.argmax(win))
    i0 = min(i0, len(y) - n)  # never overrun the clip end
    return y[i0:i0 + n], fn


def prop_far(sig, az):
    u = far_pos(az) / 2000.0
    return np.stack([frac_delay(sig, -(POS[m] @ u) / C) for m in range(M)])


def prop_near(sig, az, dist=10.0):
    d = np.linalg.norm(POS - far_pos(az, dist), axis=1)
    x = np.stack([frac_delay(sig, d[m] / C) for m in range(M)])
    return x * (dist / d)[:, None]


def quadrotor(t, seed=2):
    """Synthetic 4-rotor UAV: blade-pass harmonics (150 Hz comb) with AM."""
    r = np.random.default_rng(seed)
    y = np.zeros_like(t)
    for h in range(1, 7):
        y += np.sin(2 * np.pi * 150 * h * t + r.uniform(0, 6)) / h
    env = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 1.7 * t + r.uniform(0, 6)), -1, 1)
    return y * env


def main():
    ast, sess, yam_names = load_model_map()
    bird, bird_fn = esc_clip("chirping_birds", idx=0)
    heli, heli_fn = esc_clip("helicopter", idx=0)
    engine, eng_fn = esc_clip("engine", idx=0)
    print(f"clips: bird={bird_fn} heli={heli_fn} engine={eng_fn}", flush=True)

    print("model sanity (raw clips):", flush=True)
    for nm, clip in (("bird", bird), ("helicopter", heli), ("engine", engine)):
        sa = score_ast(ast, clip)
        sy = score_yamnet(sess, yam_names, clip)
        print(f"  {nm:11s} AST={sa[0][0][:26]}={sa[0][1]:.2f}  "
              f"YAMNet={sy[0][0][:26]}={sy[0][1]:.2f}", flush=True)

    rows = []
    for variant in ("helicopter+engine", "synthetic-quadrotor"):
        rng = np.random.default_rng(21)
        t = np.arange(int(FS * DUR)) / FS
        noise = 0.1 * rng.standard_normal((M, len(t)))
        mix = prop_far(bird * 10 ** (-10 / 20), AZ_T)
        if variant.startswith("helicopter"):
            mix = mix + prop_near(heli, AZ_I) * 10 ** (40 / 20) * 0.1 \
                / np.sqrt(np.mean(heli ** 2))
            mix = mix + prop_far(engine, 120.0) * 10 ** (30 / 20) * 0.1 \
                / np.sqrt(np.mean(engine ** 2))
            refs = (bird, heli, engine)
        else:
            uav = quadrotor(t)
            mix = mix + prop_near(uav, AZ_I, dist=50.0) * 10 ** (35 / 20) * 0.1 \
                / np.sqrt(np.mean(uav ** 2))
            mix = mix + prop_far(engine, 120.0) * 10 ** (25 / 20) * 0.1 \
                / np.sqrt(np.mean(engine ** 2))
            refs = (bird, uav, engine)
        mix = mix + noise

        for meth in ("DAS", "LCMV", "GSC", "RMS->GSC", "SubspaceNull"):
            x = gain_normalize(mix) if "GSC" in meth else mix
            if meth == "DAS":
                y = StandardDAS(POS, FS).process(mix, far_pos(AZ_T))
            elif meth == "LCMV":
                y = LCMVSMI(POS, FS).process(mix, far_pos(AZ_T))
            elif meth == "GSC":
                y = GSC_MCWF(POS, FS).process(x, far_pos(AZ_T))
            elif meth == "RMS->GSC":
                y = GSC_MCWF(POS, FS).process(x, far_pos(AZ_T))
            else:
                y = subspace_null(mix, AZ_T, k=2)
            sisdr = si_sdr(y, bird)
            m_ast = margin(score_ast(ast, y))
            m_yam = margin(score_yamnet(sess, yam_names, y))
            rows.append(dict(Variant=variant, Method=meth, SI_SDR=sisdr,
                             AST_bird_margin=m_ast, YAMNet_bird_margin=m_yam,
                             Tonality=tonality8k(y)))
            print(f"[{variant}/{meth:12s}] SI-SDR={sisdr:6.1f}  AST-margin="
                  f"{m_ast:+6.2f}  YAMNet-margin={m_yam:+6.2f}  tonal={tonality8k(y):5.1f}",
                  flush=True)
        # raw-mic reference row
        m_ast = margin(score_ast(ast, mix[0]))
        m_yam = margin(score_yamnet(sess, yam_names, mix[0]))
        rows.append(dict(Variant=variant, Method="raw-mic0", SI_SDR=si_sdr(mix[0], bird),
                         AST_bird_margin=m_ast, YAMNet_bird_margin=m_yam,
                         Tonality=tonality8k(mix[0])))
        print(f"[{variant}/raw-mic0     ] AST-margin={m_ast:+6.2f} "
              f"YAMNet-margin={m_yam:+6.2f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(REPO, "docs", "bird_aerial_benchmark.csv"),
                              index=False)
    print("\nSaved docs/bird_aerial_benchmark.csv")


if __name__ == "__main__":
    main()

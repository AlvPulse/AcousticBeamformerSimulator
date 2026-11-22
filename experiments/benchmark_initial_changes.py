"""
INITIAL SMALL CHANGES for the company system -- complete metric analysis.

Their architecture is KEPT: grid scan -> per-beam audio -> level history
-> anomaly + clustering -> detector. The menu swaps only the beamformer
and/or adds a pre-normalization (BOTH variants kept -- we do not know
which one they use internally):
  LCMV            baseline (their gold standard)
  PhaseOnly       swap
  RMS->LCMV       norm-first then LCMV (variant A)
  MAX->LCMV       norm-first then LCMV (variant B, reported company scheme)
  GSC             swap
  RMS->GSC        norm-first + GSC
  MAX->GSC        norm-first + GSC
  DAS             radical simplification reference

PART 1 (synthetic + REAL ESC-50 interferers, known target reference):
  selection err, SI-SDR, purity dSIR, tonality, line-gain spread, and the
  history-anomaly-critical metric: cross-chunk level stability with an
  interferer that appears in chunks 6-11 of 16.
PART 2 (their actual field data, no ground truth): tonality, AST class
  (single model), level stability across chunks, method agreement.
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
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_extreme_inr import (
    POS, FS, DUR, C, far_pos, frac_delay,
)
from experiments.benchmark_all_pipelines import (
    capon_map, tonality8k, gen_tonal, prop_far,
)
from experiments.benchmark_phase_robustness import (
    align, si_sdr, line_gain_spread,
)
from experiments.benchmark_pipelines import purity

M = POS.shape[0]
AZ_T, AZ_I = 40.0, -60.0
INR_DB = 30.0

MENU = ["LCMV", "PhaseOnly", "RMS->LCMV", "MAX->LCMV",
        "GSC", "RMS->GSC", "MAX->GSC", "DAS"]
NORM = {"RMS": lambda x: x / (np.sqrt(np.mean(x ** 2, axis=1, keepdims=True)) + 1e-12),
        "MAX": lambda x: x / (np.max(np.abs(x), axis=1, keepdims=True) + 1e-12)}


def esc50_clip(category, idx=0):
    meta = pd.read_csv(os.path.join(REPO, "ESC-50-master", "meta", "esc50.csv"))
    rows = meta[meta.category == category]
    fn = rows.iloc[idx].filename
    y, _ = librosa.load(os.path.join(REPO, "ESC-50-master", "audio", fn),
                        sr=FS, mono=True)
    seg = y[int(0.5 * FS):int(0.5 * FS) + int(DUR * FS)]
    if len(seg) < int(DUR * FS):
        seg = np.pad(seg, (0, int(DUR * FS) - len(seg)))
    return seg, category


def make_scene(intf_sig):
    t = np.arange(int(FS * DUR)) / FS
    rng = np.random.default_rng(7)
    tgt = gen_tonal(t) * 10 ** (-10 / 20)
    x = prop_far(tgt, AZ_T)
    e_int = np.sqrt(np.mean(intf_sig ** 2))
    if e_int > 1e-12:  # interferer-off chunks pass zeros
        x = x + prop_far(intf_sig, AZ_I) * (10 ** (INR_DB / 20) * 0.1 / e_int)
    return x + 0.1 * rng.standard_normal((M, len(t))), gen_tonal(t)


def apply_menu(name, mix):
    """Return (processed extraction fn, grid-selection fn)."""
    norm_kind = None
    if name.startswith("RMS->"):
        norm_kind = "RMS"
    if name.startswith("MAX->"):
        norm_kind = "MAX"
    core = name.split("->")[-1]
    x = NORM[norm_kind](mix) if norm_kind else mix

    if core == "LCMV":
        bf = LCMVSMI(POS, FS)
        return (lambda az: bf.process(x, far_pos(az)),
                lambda: capon_map(x, kind="lcmv"))
    if core == "PhaseOnly":
        bf = PhaseOnlyLCMV(POS, FS)
        return (lambda az: bf.process(x, far_pos(az)),
                lambda: capon_map(x, kind="phaseonly"))
    if core == "GSC":
        bf = GSC_MCWF(POS, FS)
        return (lambda az: bf.process(x, far_pos(az)),
                lambda: _sweep(bf, x))
    if core == "DAS":
        bf = StandardDAS(POS, FS)
        return (lambda az: bf.process(x, far_pos(az)),
                lambda: _sweep(bf, x))
    raise ValueError(name)


def _sweep(bf, x):
    azs = np.arange(-180, 180, 10.0)
    p = np.array([np.mean(bf.process(x, far_pos(a)) ** 2) for a in azs])
    return azs, p


def ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def part1():
    classes = ["dog", "engine", "rain", "wind", "chainsaw", "car_horn"]
    out_csv = os.path.join(REPO, "docs", "initial_changes_synthetic.csv")
    rows = []
    done = set()
    if os.path.exists(out_csv):  # resume: never lose a completed class
        df_prev = pd.read_csv(out_csv)
        rows = df_prev.to_dict("records")
        done = set(df_prev[df_prev.Part == "synthetic"].Class.unique())
    for cls in classes:
        if cls in done and not df_prev[(df_prev.Class == cls) &
                                       (df_prev.Part == "stability")].empty:
            continue
        intf, _ = esc50_clip(cls)
        # --- selection + extraction-quality (interferer ON, 1 scene) ---
        mix, ref = make_scene(intf)
        for name in MENU:
            extract, sel_fn = apply_menu(name, mix)
            azs, pows = sel_fn()
            sel = float(azs[np.argmax(pows)])
            y_or = extract(AZ_T)
            ya, _ = align(y_or, ref)
            sir_o, _, _ = purity(y_or, (ref, intf))
            sir_in, _, _ = purity(mix[0], (ref, intf))
            rows.append(dict(Part="synthetic", Class=cls, Method=name,
                             SelErr=ang(sel, AZ_T),
                             SI_SDR=si_sdr(y_or, ref), dSIR=sir_o - sir_in,
                             Tonality=tonality8k(y_or),
                             LineSpread=line_gain_spread(y_or, ref),
                             OnsetShift=np.nan, QuietScatter=np.nan))
            print(f"[{cls}/{name}] selErr={rows[-1]['SelErr']:5.1f} "
                  f"SI-SDR={rows[-1]['SI_SDR']:6.1f} dSIR={rows[-1]['dSIR']:6.1f} "
                  f"tonal={rows[-1]['Tonality']:5.1f} lineSpread={rows[-1]['LineSpread']:4.1f}",
                  flush=True)
        # --- level stability: 16 chunks, interferer in chunks 6-11 ---
        for name in [m for m in MENU if m != "DAS" and m != "PhaseOnly"]:
            levels = []
            for k in range(16):
                on = 6 <= k <= 11
                mix_k, ref_k = make_scene(intf if on else np.zeros_like(intf))
                extract, _ = apply_menu(name, mix_k)
                y = extract(AZ_T)
                levels.append(10 * np.log10(np.mean(y ** 2) + 1e-20))
            levels = np.array(levels) - levels[0]
            base = 1.4826 * np.median(np.abs(levels[:6] - np.median(levels[:6]))) + 1e-6
            z = np.abs(levels - np.median(levels[:6])) / base
            rows.append(dict(Part="stability", Class=cls, Method=name,
                             SelErr=np.nan, SI_SDR=np.nan, dSIR=np.nan,
                             Tonality=np.nan, LineSpread=np.nan,
                             OnsetShift=float(levels[6]),
                             QuietScatter=float(np.std(np.concatenate([levels[:6], levels[12:]]))),
                             FalseAnom=int(np.sum(z[6:] > 3))))
            print(f"[{cls}/{name} STABILITY] onset={levels[6]:7.2f} dB "
                  f"quiet={rows[-1]['QuietScatter']:5.2f} dB falseAnom={rows[-1]['FalseAnom']}/10",
                  flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


def part2():
    print("\nPART 2: their field data (no ground truth)")
    from transformers import pipeline
    ast = pipeline("audio-classification",
                   model="MIT/ast-finetuned-audioset-10-10-0.4593")
    base = sorted(glob.glob(os.path.join(REPO, "data", "field_tests", "*", "*.wav")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    chunks = [p for p in base if os.path.basename(p) in
              ("1.wav", "10.wav", "20.wav", "30.wav", "40.wav")]
    rows = []
    for d in (8.0, 22.0, -52.0, 172.0):
        for name in MENU:
            levels, ast_top, ast_conf, tons = [], None, 0.0, []
            for p in chunks:
                x, _ = sf.read(p, dtype="float64")
                x = x.T
                extract, _ = apply_menu(name, x)
                y = extract(d)
                levels.append(10 * np.log10(np.mean(y ** 2) + 1e-20))
                tons.append(tonality8k(y))
                y16 = librosa.resample(np.asarray(y).flatten(), orig_sr=FS,
                                       target_sr=16000)
                y16 = y16 / (np.max(np.abs(y16)) + 1e-9) * 0.9
                r = ast({"raw": y16, "sampling_rate": 16000}, top_k=1)[0]
                ast_top = r["label"]
                ast_conf = max(ast_conf, r["score"])
            rows.append(dict(Part="field", Dir=d, Method=name,
                             LevelStd_dB=float(np.std(levels)),
                             Tonality=float(np.median(tons)),
                             AST_top=ast_top, AST_conf=round(ast_conf, 2)))
            print(f"[field az={d:4.0f} {name:11s}] levelStd={rows[-1]['LevelStd_dB']:5.2f} dB "
                  f"tonal={rows[-1]['Tonality']:5.1f} AST={ast_top}({ast_conf:.2f})",
                  flush=True)
    return rows


def main():
    rows = part1()
    df1 = pd.DataFrame(rows)
    df1.to_csv(os.path.join(REPO, "docs", "initial_changes_synthetic.csv"), index=False)
    print("\nSYNTHETIC SUMMARY (mean over 6 ESC-50 interferer classes):")
    g = df1[df1.Part == "synthetic"].groupby("Method").mean(numeric_only=True)
    print(g[["SelErr", "SI_SDR", "dSIR", "Tonality", "LineSpread"]].to_string(
        float_format=lambda v: f"{v:7.2f}"))
    gs = df1[df1.Part == "stability"].groupby("Method").mean(numeric_only=True)
    print("\nSTABILITY SUMMARY (history-anomaly impact; onset shift should "
          "equal pure leakage only):")
    print(gs[["OnsetShift", "QuietScatter", "FalseAnom"]].to_string(
        float_format=lambda v: f"{v:7.2f}"))

    rows2 = part2()
    df2 = pd.DataFrame(rows2)
    df2.to_csv(os.path.join(REPO, "docs", "initial_changes_field.csv"), index=False)


if __name__ == "__main__":
    main()

"""
REAL-WORLD STFT showcase: LCMV-replacement candidates on their field data.

For representative chunks and the strongest persistent directions:
  - INPUT: raw mic-0 spectrogram (what enters every method)
  - OUTPUT: spectrogram of each method's extracted audio
    (LCMV baseline | PhaseOnly | GSC | GSC+gn | SubspaceNull | DAS)
  plotted side-by-side with a SHARED dB scale so artifacts are visible.

Prior/posterior analysis per method:
  - LSD(input, output): total spectral modification the method applied
  - tonality in->out, band-power shift (250-1100 Hz vs 1500-3500 Hz)
  - AST class on the raw mic (prior) vs on each output (posterior)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator.beamformers.das import StandardDAS
from simulator.beamformers.lcmv_smi import LCMVSMI
from simulator.beamformers.robust_phase import PhaseOnlyLCMV
from simulator.beamformers.gsc_mcwf import GSC_MCWF
from experiments.benchmark_extreme_inr import POS, FS, far_pos, subspace_null
from experiments.benchmark_all_pipelines import gain_normalize, tonality8k

DIRS = [8.0, -52.0]
CHUNKS = ["1.wav", "20.wav", "40.wav"]
METHODS = ["LCMV", "PhaseOnly", "GSC", "GSC+gn", "SubspaceNull", "DAS"]


def stft_db(x, nper=512):
    f, t, Z = spsig.stft(np.asarray(x).flatten(), fs=FS, nperseg=nper,
                         noverlap=nper - 256, window="hann")
    return f, t, 20 * np.log10(np.abs(Z) + 1e-10)


def band_power_db(x, lo, hi):
    f, P = spsig.periodogram(np.asarray(x).flatten(), FS, nfft=16384,
                             window="blackman")
    sel = (f >= lo) & (f <= hi)
    return 10 * np.log10(np.sum(P[sel]) + 1e-20)


def main():
    from transformers import pipeline
    ast = pipeline("audio-classification",
                   model="MIT/ast-finetuned-audioset-10-10-0.4593")

    das = StandardDAS(POS, FS)
    lcmv = LCMVSMI(POS, FS)
    po = PhaseOnlyLCMV(POS, FS)
    gsc = GSC_MCWF(POS, FS)

    base = sorted(glob.glob(os.path.join(REPO, "data", "field_tests", "*", "*.wav")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    outdir = os.path.join(REPO, "docs", "field_stft")
    os.makedirs(outdir, exist_ok=True)
    rows = []

    def ast_top(y):
        y16 = librosa.resample(np.asarray(y).flatten(), orig_sr=FS,
                               target_sr=16000)
        y16 = y16 / (np.max(np.abs(y16)) + 1e-9) * 0.9
        r = ast({"raw": y16, "sampling_rate": 16000}, top_k=1)[0]
        return f"{r['label']}({r['score']:.2f})"

    for cname in CHUNKS:
        path = [p for p in base if os.path.basename(p) == cname][0]
        x, _ = sf.read(path, dtype="float64")
        x = x.T  # (64, n)
        x_in = x[0]  # mic 0 = the prior
        prior = dict(ast=ast_top(x_in), tonal=tonality8k(x_in),
                     low=band_power_db(x_in, 250, 1100),
                     high=band_power_db(x_in, 1500, 3500))
        print(f"[{cname}] PRIOR (mic0): AST={prior['ast']} tonal={prior['tonal']:.1f}",
              flush=True)

        for d in DIRS:
            outs = {
                "LCMV": lcmv.process(x, far_pos(d)),
                "PhaseOnly": po.process(x, far_pos(d)),
                "GSC": gsc.process(x, far_pos(d)),
                "GSC+gn": gsc.process(gain_normalize(x), far_pos(d)),
                "SubspaceNull": subspace_null(x, d, k=2),
                "DAS": das.process(x, far_pos(d)),
            }
            # ---- figure: input + 6 outputs, shared dB scale ----
            fig, axs = plt.subplots(len(METHODS) + 1, 1, figsize=(9, 3.2 * (len(METHODS) + 1)),
                                    sharex=True)
            f0, t0, S0 = stft_db(x_in)
            ref_db = S0
            vmin, vmax = np.percentile(ref_db, 5), np.percentile(ref_db, 99)
            im = axs[0].pcolormesh(t0, f0, S0, vmin=vmin, vmax=vmax,
                                   shading="auto", cmap="magma")
            axs[0].set_ylabel("Hz")
            axs[0].set_title(f"{cname[:-4]} az={int(d)} -- INPUT mic0 "
                             f"(AST {prior['ast']})", fontsize=9)
            for i, m in enumerate(METHODS, start=1):
                fm, tm, Sm = stft_db(outs[m])
                axs[i].pcolormesh(tm, fm, Sm, vmin=vmin, vmax=vmax,
                                  shading="auto", cmap="magma")
                axs[i].set_ylabel("Hz")
                axs[i].set_title(f"OUTPUT {m}", fontsize=9)
            axs[-1].set_xlabel("time (s)")
            fig.suptitle(f"Field {cname[:-4]} @ az {int(d)}: input vs LCMV-replacement "
                         f"outputs (shared dB scale)", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(os.path.join(outdir, f"{cname[:-4]}_az{int(d)}_stft.png"),
                        dpi=110)
            plt.close(fig)

            # ---- prior/posterior numbers ----
            f_in = stft_db(x_in)[2]
            for m in METHODS:
                y = outs[m]
                f_y = stft_db(y)[2]
                n = min(f_in.shape[1], f_y.shape[1])
                lsd = float(np.mean(np.abs(f_in[:, :n] - f_y[:, :n])))
                rows.append(dict(Chunk=cname, Az=d, Method=m,
                                 LSD_in_out=round(lsd, 1),
                                 Tonality_in=prior["tonal"],
                                 Tonality_out=round(tonality8k(y), 1),
                                 LowBand_shift=round(band_power_db(y, 250, 1100)
                                                     - prior["low"], 1),
                                 HighBand_shift=round(band_power_db(y, 1500, 3500)
                                                      - prior["high"], 1),
                                 AST_prior=prior["ast"], AST_post=ast_top(y)))
                r = rows[-1]
                print(f"  [{cname[:-4]}/az{int(d)}/{m:12s}] LSD={r['LSD_in_out']:5.1f} "
                      f"tonal {r['Tonality_in']:.0f}->{r['Tonality_out']:.0f} "
                      f"low {r['LowBand_shift']:+6.1f} high {r['HighBand_shift']:+6.1f} "
                      f"AST {r['AST_prior']} -> {r['AST_post']}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(REPO, "docs", "field_stft_analysis.csv"),
                              index=False)
    print(f"\nSaved docs/field_stft_analysis.csv + spectrograms in {outdir}")


if __name__ == "__main__":
    main()

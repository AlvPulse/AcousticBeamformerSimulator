# Metrics Glossary (plain language)

Every number reported in this project, what it means, its units, which way
is good, and — importantly — what it CANNOT tell you. Synthetic metrics
need the clean source recordings (ground truth); field metrics do not.

---

## A. Metrics that need ground truth (synthetic tests only)

### SI-SDR (dB) — "how close is the output to the clean target?"
Scale-Invariant Signal-to-Distortion Ratio. We take the extracted audio,
time-align it to the clean target recording (cross-correlation), fit the
best amplitude scale, then compare: energy of the target vs energy of
everything else (interferers + noise + distortion) in the output.
**Higher = better. 0 dB = error as loud as the target. 14 dB ≈ 4% error
energy.** Because it is scale-invariant and lag-aligned, absolute volume
and bulk delay don't matter.
Limitation: says nothing about WHERE the error is (noise? interferer?
spectral distortion?) — that's what purity and LSD are for.

### Purity / ΔSIR (dB) — "is the output the source we pointed at?"
Least-squares decomposition: we take the clean recordings of every source
(target AND interferers), align each to the output, and fit
`output ≈ a·target + b·interferer1 + c·interferer2`. Then:
- **target retention** = energy explained by the target coefficient
- **interference residual** = energy explained by interferer coefficients
- **ΔSIR** = (target/interferer ratio at output) − (same ratio at a raw
  mic) = how much better the beamformer made the ratio. **Higher = better;
  negative = output is more interferer than target.**
Limitation: the alignment step breaks when the error is ≫ the target
(returns NaN/degenerate values). In extreme regimes we use tonal-SNR
instead (below).

### Line-gain spread (dB) — "does the filter color the target's spectrum?"
Gain the output applies to each of the target's harmonic lines
(350/700/1050 Hz), computed as 10·log10(P_out/P_clean) per line; the
metric is max−min across lines. **Lower = better (flat = faithful).**
DAS ≈ 0.3 dB at every error level; LCMV 2.5–3 dB calibrated, 8–10 dB with
phase error — this is the per-frequency-bin coloration mechanism, measured.

### LSD (dB) — log-spectral distance between output and clean-target
spectrograms (mean |log-magnitude difference| over 200–3000 Hz).
Lower = better. In noisy extraction regimes it is dominated by the noise
floor between harmonics, so we report it mainly for the fidelity question.

## B. Metrics that do NOT need ground truth (used on synthetic AND field)

### Tonal SNR at known lines (dB) — "is the tonal target detectable?"
Power at the target's harmonic lines vs power at nearby OFF-line control
frequencies (same bandwidth). The control bands estimate the local noise +
interference floor, so no clean reference is needed. **Higher = the target
stands out more.** Used as: purity proxy in extreme-interference regimes,
and the improvement column = output value minus raw-mic value.

### Tonality / harmonic contrast (dB) — "how tonal is this audio?"
Peak-to-median ratio of the time-averaged power spectrum (200–2500 Hz).
Tonal sources (sirens, drones) score 25–90 dB; broadband noise scores
2–14 dB. Used to (a) choose which beamformer output to keep, (b) decide
whether to apply the comb filter, (c) as a proxy for LAM response quality
(a classifier keys on exactly this structure). No reference needed.

### DOA hit rate (%) — "did the localizer find the true direction?"
Fraction of runs where an estimated azimuth is within 10–15° of a true
source azimuth. On field data (no truth) we replace truth with
**cross-estimator agreement**: NormMUSIC vs SRP vs OverIVA-manifold all
estimating the same angle independently (measured 1.2° mean on the field
chunks).

### Selection accuracy (%) — "did the pipeline pick the right beam/source?"
In blind tests: fraction of draws where the pipeline's chosen output
(beam × spectral track) belongs to the intended target class (checked
against truth in synthetic; on field data replaced by persistence +
tonality + LAM agreement).

### RTF — real-time factor
Seconds of compute per second of audio, same machine. 0.15 RTF = 6.7×
faster than real time. Lower = cheaper. Not comparable across machines.

## C. The legacy metric we RETIRED

`harmonic_score` (track_harmonics max_score): scored the CLEAN target at
−74 dB and a pure tone at −150; adding noise IMPROVED it. It ranks outputs
by broadband-noise leakage, not target quality. Do not use for cross-method
ranking until recalibrated (open item). Its components are still fine for
per-frame peak tracking.

## D. Success criteria used in Monte Carlos (explicit, per draw)

- DOA_siren/bark/urban: some candidate azimuth within 15° of truth
- siren_recovered: an output whose tracked f0 matches a siren harmonic AND
  tonal-SNR margins > 10 dB vs each other source's characteristic band
- bark_emitted: a broadband output within 15° of the bark azimuth
- urban_attributed: best-margin urban output's beam within 15° of truth
- spurious: outputs matching no source
Confidence intervals: Wilson 95%.

# Pipeline Specification (exact, stage by stage)

The deployed pipeline ("v5") and how it is tested. Every stage lists its
input, output, the exact methods used, and where each number in the
reports comes from. Nothing here is aspirational — each stage exists in
code under `experiments/`.

```
 64-ch wav chunk (2.12 s, 8 kHz)
        │
 [S0] Channel QC & gain normalization        (per chunk)
        │
 [S1] Candidate DOA  ──────────── angles only (no audio)
        │
 [S2] Per-candidate extraction ── one audio stream per angle
        │
 [S3] Spectral decomposition ─── multi-f0 + broadband emission:
        │                          several PURE sounds per angle
 [S4] Weak-tonal integration ──── multi-chunk narrowband (static-
        │                          continuous interference case)
 [S5] Track persistence ───────── candidates -> tracks across chunks
        │
 [S6] Detection/LAM ───────────── your classifier judges the
                                 candidate audio, never a grid argmax
```

## [S0] Channel QC + normalization
- **Input:** raw 64-ch chunk. **Output:** cleaned chunk + per-channel gain
  estimate.
- Measured on your field data: 8.1 dB per-channel level spread, no dead or
  clipping channels (docs/field_quick_pass.csv).
- **Gain normalization (the exact formula):**
  `g_m = sqrt(mean_t x_m(t)^2)` over the chunk; `x_m <- x_m / g_m`.
  This equalizes each channel's total broadband power.
  **Assumption and its limit:** if all channels heard the same far-field
  mixture, `g_m` IS the mic gain. When a strong source is near-field or
  directional, channels genuinely receive different levels, so `g_m`
  conflates hardware gain with acoustic level — normalization then
  *imposes* the far-field assumption instead of measuring gains.
  Validation: with synthetic per-mic gains U(0.5,2) this recovers
  extraction SI-SDR from 0.7 to 10.2 dB (docs/mc_benchmark.csv). A
  coherence-weighted gain estimator (gain ratios only in bands/channels
  with high cross-coherence) is the more honest field version — open item,
  small.

## [S1] Candidate DOA (angles only)
- **Input:** normalized chunk STFT (512/256, hann). **Output:** K=3–4
  azimuths.
- Methods: **NormMUSIC** (primary; pyroomacoustics) and **SRP-PHAT**
  (secondary), 250–1100 Hz (your array aliases above ~1.2 kHz), 2° grid.
- No audio is produced here. Gating (quietest-50% frames) is applied
  OPPORTUNISTICALLY: never required, only to sharpen.
- Robustness measured: 100% hit at gain spread ≤12 dB, phase error ≤±4 cm
  delay / ±45° all-pass; movers benign; static-continuous +50 dB is the
  one failing regime → handled at [S4].

## [S2] Per-candidate extraction (interchangeable — no target known)
- **Input:** chunk + one candidate azimuth. **Output:** one audio stream.
- Three extractors compete per beam; the winner is chosen by measured
  tonality AFTER [S3] trial ("full-path scoring"):
  1. **DAS** — fixed weights, spectrally flat, cheap; best for the
     dominant broadband source.
  2. **GSC-MCWF (+gain-norm)** — default on uncalibrated arrays:
     phase-robust (12.2 dB SI-SDR at ±20° phase error vs 1.3 for LCMV),
     spectrally flat (line-gain spread 0.2–1.2 dB).
  3. **SubspaceNull(k=2)/LCMV-null-others** — for extreme INR (+40–50 dB)
     where GSC leaks; needs the array to be roughly calibrated.
- This is the "interchangeable" part you asked for: every candidate angle
  gets every extractor; data (not assumptions) picks.

## [S3] Spectral decomposition (multi-f0 + broadband)
- **Input:** one extracted stream. **Output:** up to 2 tonal tracks
  (f0 + combed audio) + optionally 1 broadband track.
- f0 tracking: peak-picking in 150–1100 Hz scored by harmonic-sum vs local
  background; comb filter at f0×h (±25 Hz, floor −26 dB). After locking
  f0₁ we notch it and re-track for a second tonal source in the same beam
  (two pure sounds can share one angle).
- Broadband emission: rank beams by absolute energy surviving all tonal
  notches; top beam(s) = broadband source(s).
- **Known limitation (open):** greedy tracking fails when a louder tonal
  masks a weaker one in the same beam — multi-pitch estimator is the fix.

## [S4] Multi-chunk narrowband integration (weak tonal under static-
      continuous interference)
- **Input:** many consecutive chunks + a recovered f0. **Output:** the
  weak target's angle, by coherent averaging across chunks (≈ +18 dB for
  60 chunks). Prior used: the TARGET persists — allowed; no interferer
  priors. Status: designed, validation pending (this is the remaining
  synthetic gap; see gaps list).

## [S5] Track persistence
- **Input:** per-chunk candidates/outputs. **Output:** tracks with DOA
  stability, persistence counts, per-candidate feature time series —
  exactly the "history" features your existing anomaly/clustering stage
  consumes (per candidate instead of per grid node).
- Field preview: persistent directions at ~172°, ~8°, ~22°, ~−52° across
  the 52 chunks.

## [S6] Detection / LAM
- **Input:** candidate WAVs. **Output:** class scores per candidate.
- The detector picks the target among candidates — never energy argmax,
  never LAM on a full grid. (AST stack now installed; A/B test pending.)

---

## How combinations are tested (the exact procedure)

1. **Scene:** 64-mic real geometry, exact fractional-delay synthesis;
   sources: tonal target (350/700/1050 Hz), broadband interferer
   (near-field 10 m or far), urban hum (430/860 Hz). Knobs randomized per
   draw: azimuths, envelopes, burst timing, INR (+40/45/50 dB), gain
   spread (0/6/12 dB), phase/delay errors, wind, movement.
2. **Draws:** 10–20 per condition; seeds fixed; every system runs on the
   SAME scene draw (paired comparison).
3. **Systems compared end-to-end:** A (grid LCMV + energy peak — company
   baseline), A′ (grid PhaseOnly), A+ (grid + gain-norm), B (S1→S2→S3→S6),
   C (OverIVA blind separation + manifold DOA), plus per-stage ablations.
4. **Success criteria + metrics:** see METRICS.md section D; Wilson CIs.
5. **Evidence files:** docs/{mc_benchmark, pipeline_benchmark,
   extreme_inr_benchmark, blind_mc_results, industrial_benchmark,
   phase_robustness, field_quick_pass}.csv

## Reporting field results WITHOUT ground truth (the protocol)

No clean references exist for field recordings, so field reports contain
ONLY reference-free evidence, explicitly labeled:
1. **Cross-estimator agreement** — NormMUSIC vs SRP vs OverIVA-manifold
   estimating the same angle (independent physics agreeing). Field pass 1:
   1.2° mean MUSIC-vs-SRP.
2. **Persistence** — a candidate that survives N chunks at a stable angle
   vs one-chunk ghosts; the 4 persistent directions found.
3. **Tonality improvement** — tonal-SNR of extracted audio vs the raw mic
   channel (dB gain, no reference needed).
4. **LAM confidence delta** — AST score on candidate audio vs AST score
   on raw channels / grid-scan audio (the detector's own opinion).
5. **Physical plausibility** — elevation ≥ 0, smooth tracks, speeds.
6. Anything resembling a "detection rate" on field data will be produced
   ONLY by your detector on frozen candidate sets — never by us claiming
   truth we don't have.

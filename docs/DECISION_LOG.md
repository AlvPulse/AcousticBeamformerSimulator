# Decision & Pivot Log

Engineering log of steps, findings, and direction changes for the beamforming
pipeline evaluation. Sources of evidence are referenced per entry. Dates:
2026-11-16 to 2026-11-18.

---

## 1. 2026-11-16 — Repository & remote audit

**Step:** Analyzed local repo + fetched remote branches. Found local branch
11 commits ahead; remote added MUSIC/SRP-PHAT/INCM-MVDR, full-pipeline
orchestrator, Persian docx reports, and a Z-coordinate geometry fix.

**Pivot:** none yet, but flagged that remote docs/reports would need
regeneration after later findings (see #4).

## 2. 2026-11-16 — External 3-pipeline plan reviewed

**Step:** Reviewed the ChatGPT A/B/C pipeline plan (grid-LCMV baseline /
DOA-first + robust LCMV / overdetermined BSS) against the repo.

**Decision:** Adopt the A/B/C framing for all comparisons; reject FRIDA as a
central method (later confirmed broken in pyroomacoustics for multi-source);
keep gen-1 `acoustic_sim` (physically correct, unit-tested), rebuild gen-2
evaluations.

## 3. 2026-11-16 — QC finds the steering conjugation bug (CRITICAL PIVOT)

**Step:** Code QC across `simulator/beamformers/`. Found `base.py` returned
conj(a_phys) — every geometry-trusting beamformer steered at the MIRROR of
the look direction. Proven empirically: scene with sources at +40/-60 deg,
repo DAS peaked at +120 deg (mirror of the louder source); pyroomacoustics
SRP/NormMUSIC on identical data returned 38-40/-58-60.

**Pivot:** ALL pre-fix benchmark conclusions invalid. The documented
"standard LCMV collapses, data-driven methods win" story was an artifact:
LCMV was nulling the target because it thought the target was a
mirror-image interferer. Evidence: `experiments/qc_steering_null_check.py`.

**Also found:** CMT taper ~ identity (RobustLCMV == LCMV numerically);
integer-sample scene delays (~0.8 rad phase error @2 kHz); harmonic_score
metric pathology (clean target scores -74 dB, pure tone -150 dB, adding
noise IMPROVES score -> ranks methods by noise leakage, not quality);
pyroomacoustics `auxiva(init_eig=True)` broken for overdetermined input.

## 4. 2026-11-16 — Fixes applied, tests added, benchmarks re-run

**Step:** Fixed steering sign, exact fractional-delay synthesis,
frequency-dependent CMT, vectorized Ledoit-Wolf, eigh-based stable
inversion. Added 4 gen-2 regression tests (12/12 pass; these would have
caught the bug). Re-ran all three legacy benchmarks.

**Result:** LCMV family SI-SDR improved ~25-40 dB in synthetic scenarios
(e.g. S3 wind: -45.6 -> -4.6 dB); cGMM-MVDR emerged as top extractor;
LCMV-grid IRR 8.4 -> 13.9 dB. Calibration-breakdown table remains
uninterpretable (metric pathology + aliased 1.0 m-radius geometry).

**Pivot:** treat calibration table as void; method ranking re-derived.

## 5. 2026-11-16 — Unified benchmark establishes the correct architecture

**Step:** Built `benchmark_unified.py`: Stage A DOA estimators (angles only),
Stage B grid beamformers (directional audio), Stage C overdetermined
OverIVA + per-source manifold DOA. Fixed prior interface flaws (DOA methods
forced to emit audio; BSS forced to single output with Azimuth=0
placeholders).

**Result:** NormMUSIC 0 deg DOA error on all 3 sources; LCMV oracle
extraction 14.2 dB SI-SDR; OverIVA target 10.4 dB with 4 deg manifold DOA.
Energy-argmax always lands on the loudest interferer -> target selection is
architectural, not algorithmic.

## 6. 2026-11-16 — Gain imbalance integrated (6 dB / 12 dB)

**Step:** Added per-mic gain spread conditions to the unified benchmark.

**Result:** DOA stage immune (<=1-8 deg error at 12 dB spread). LCMV
extraction collapses (14.2 -> 0.7 dB); PhaseOnly/GSC degrade gracefully.

## 7. 2026-11-17 — Phase 0 executed (hygiene)

**Step:** Branch `fix/steering-and-benchmarks`, commits + push, PR prepared,
requirements.txt, logs gitignored, LCMV pinv-stall guard (outlier RTF 7813
never recurred). Remote docs NOT merged (pre-fix, contaminated).

## 8. 2026-11-17 — Phase 1: gain normalization solves uncalibrated arrays

**Step:** 20-draw Monte Carlo x 3 gain levels x {anechoic, wind+RPM-drift}
scenes; new tonality selector (peak-to-median of mean spectrum) replacing
broken harmonic_score (selection 67% -> 100%).

**Result (PIVOT to B3):** NormMUSIC -> GSC + per-channel RMS normalization:
10.2 dB SI-SDR at +/-6 dB/mic (vs 0.7 LCMV), 100% selection in 90 draws,
RTF 0.16 vs 1.5-3.4 for grid scans. A/A'/A+ (grid+auto-cal) all 0% selection
in 60 draws -> "keep our pipeline, swap/add X" is empirically dead.
Evidence: `docs/mc_benchmark.csv`.

**Also:** SRP/NormMUSIC exact (<=1 deg) on real 64-mic geometry in the
alias-safe 250-1100 Hz band. FRIDA multi-source unsupported (basis-size
assert) -> dropped.

## 9. 2026-11-18 — Regime pivot: extreme INR ("close dog vs 2 km siren")

**Challenge:** 10 dB rejection is meaningless for near interferer (+46 dB
spreading, plus absorption) vs distant target. All prior scenes had +4 dB
interferers — wrong regime.

**Step:** `benchmark_extreme_inr.py`: bark at 10 m near-field +40/+50 dB,
siren at 2 km 20 dB BELOW sensor noise, real 64-mic geometry.

**Result (cascade adopted):** spatial nulling alone ~20 dB; harmonic-comb
post-filter adds 16-21 dB -> 36-42 dB total improvement (separated);
worst-case same-beam: comb alone 24-31 dB. SubspaceNull exploits near-field
curvature (+5 dB even co-azimuth). GSC FAILS at extreme INR (blocking-matrix
leak) despite being the Phase-1 winner -> extractor is regime-dependent.
LS purity metric degenerates when error >> target -> replaced by tonal
band-power accounting. Evidence: `docs/extreme_inr_benchmark.csv`.

**Pivot:** pipeline gains stage B4 (adaptive post-filter); extractor
selection becomes regime-dependent.

## 10. 2026-11-18 — Blind (no-oracle) extraction: user-predicted comb failure confirmed, then fixed by architecture

**Step:** `benchmark_blind_extraction.py`: no oracle steering, no oracle f0;
urban hum (+25 dB, own tonal lines) added as third source from a third
direction.

**First run FAILED exactly as predicted:** hum leaked into every beam,
tracker locked 430 Hz everywhere, "siren" output -38.7 dB against hum.
A fixed comb is hopeless in diverse urban noise.

**Fix (architectural):** per-beam full-path extractor scoring (DAS /
SubspaceNull / LCMV-null-others, chosen after track+comb), adaptive f0
tracking, multi-f0 emission per beam (notch strongest, re-track), broadband
emission when notch-surviving energy dominates.

**Result:** blind DOA exact; az-40 beam emits BOTH urban@430 and siren@350;
siren purity +35 dB over bark, +48 dB over hum; bark emitted as broadband.
Evidence: `docs/blind_extraction_benchmark.csv`.

**Open failure modes (tracked):**
- Aligned-hum scene: greedy f0 tracking cannot find the siren under the
  hum after spatial merge -> needs multi-pitch (harmonic-consistency)
  estimator.
- Leaked tonals mis-attributed to wrong beams -> resolve via cross-chunk
  track persistence, not per-chunk argmax.
- Spurious MUSIC candidates emit junk outputs -> needs peak gating.

## 11. 2026-11-18 — Statistical validation of the blind pipeline

**Rationale:** one draw x two scenes is not decision-grade. 10 draws per
condition, randomized azimuths/envelopes/bark-timing/INR; explicit success
criteria. `experiments/benchmark_blind_mc.py`, `docs/blind_mc_results.csv`.

**Correction found during MC:** the blind scene builders divided the
near-field interferer by 4*pi*d (~22 dB at 10 m) without compensation, so
"A45/A50" actually contained bark at ~+3..+8 dB and the urban source
(+25 dB) was the true dominant. Fixed in all three builders; extreme-INR
benchmark (entry 9) was honestly labeled all along (its final normalization
cancels the 1/r factor -- rerun produced identical numbers).

**Results at TRUE +45/+50 dB INR (post-fix):**
- bark (loud near-field broadband): DOA 100%, emitted 100%, urban
  attribution 100% -- loud sources are fully solved blind
- siren audio spectrally recovered 50-70% (median margins 20-32 dB) via
  multi-f0 emission, BUT per-chunk NormMUSIC DOA of the weak siren
  collapses to 0-10% at true +50 dB -- the spatial stage cannot see a
  target 70 dB below the interferer in a single chunk
- aligned-hum scene: siren recovery 60% (spatial merge, spectral split)

## 12. 2026-11-18 — Interference-gated DOA closes the weak-target gap

**Finding:** the bark is intermittent (~24% duty). Simple energy gating
(keep the quietest 50% of frames -- no oracle) restores the siren's DOA:
gated MUSIC at 350+/-25 Hz AND wideband both localize the siren to <1 deg
under a true +50 dB interferer (validated on the worst A50 draw).

**Pivot (final architecture v4):** intermittent loud interferers are
handled by WHEN to look (energy-gated DOA + gated covariance estimation),
not only WHERE (nulls) and WHAT (combs):
  gated NormMUSIC candidates -> per-beam full-path extraction
  (DAS/SubspaceNull/LCMV-null) -> adaptive multi-f0 + broadband emission
  -> gated re-localization of recovered tonals -> cross-chunk track
  persistence -> detector.

**Remaining open items:** gated-DOA needs MC validation across draws;
multi-pitch estimator for the aligned-hum case; peak gating for spurious
MUSIC candidates (median 1-2 per chunk).

---

## Current pipeline (v4)

Gated NormMUSIC candidates -> per-beam full-path extraction (DAS /
SubspaceNull / LCMV-null-others) -> adaptive multi-f0 + broadband emission
-> gated re-localization of recovered tonals -> cross-chunk track
persistence (Phase 2) -> detector/classifier.

Extractors by regime: GSC+gain-norm (moderate imbalance, Phase-1 MC);
LCMV+gn / SubspaceNull (extreme INR); DAS (dominant broadband source).

## 13. 2026-11-20 — Industrial robustness: user's no-priors critique validated with data

**Critique tested:** burstiness-gating assumes intermittent interferers;
an industrial system must be interferer-agnostic (continuous machines,
passing cars, wind/rain, gain AND phase miscalibration).

`benchmark_industrial.py` (5 draws x {cal, miscal: gain +/-3 dB + per-mic
delay +/-1 cm}; miscalibration applied to the raw recordings):

- C50 continuous machine +50 dB: gating collapses to 0-20% (as predicted);
  ALL per-chunk localizers fail on the weak target (MUSIC wide/nb 0%,
  SRP_nb 20%, SubMUSIC_nb 20-40%). Extraction at known az holds at ~34 dB
  margins, unaffected by miscalibration.
- C50W +wind: margins halve to ~17 dB (still positive); DOA unchanged.
- M45 MOVING car +45 dB: 100% weak-target DOA for all MUSIC variants and
  even gating -- movers smear across steering directions and are BENIGN
  for stationary-target localization. Miscalibration: no measurable cost.

**Conclusions (pipeline v5):**
1. Gating is demoted to an opportunistic accelerator, not a dependency.
2. The static-continuous-extreme interferer is the only unsolved per-chunk
   case for weak-target DOA -> must be answered by MULTI-CHUNK narrowband
   integration (60 chunks/session gives ~+18 dB coherent gain; a
   target-class prior, not an interferer prior).
3. Extraction stage is calibration- and semi-wind-robust as-is.

## Next decision gates

1. Gated-DOA MC validation (expect siren DOA back to ~100%): if confirmed,
   proceed to Phase 2 field pass with pipeline v4.
2. Multi-pitch estimator for the aligned-hum case (currently 40-60%).
3. Field Phase 2 (Gate 2 in EXECUTION_PLAN.md) remains the decisive
   company evidence.


## 14. 2026-11-21 — LAM stack restored; real AST A/B confirms the coloration hypothesis

The .venv had TensorFlow 2.21 but not transformers; installed transformers
5.23 + CPU torch 2.15 (AST model ships PyTorch weights). First real
AST A/B on the same scene's target (top class + score):
  clean siren      -> Chirp tone 0.64 / Sine wave 0.32 (tonal, confident)
  GSC  @20deg err  -> Siren 0.084  (only method keeping the siren identity)
  LCMV calibrated  -> Wind instrument 0.14 (coloration shifts the class!)
  LCMV @20deg err  -> Waterfall 0.13 (tonal identity lost entirely)
  DAS  @20deg err  -> Tools 0.17 (contaminated mixture, wrong classes)
The user's per-bin-coloration concern is now classifier-verified: LCMV
changes what LAM hears; GSC preserves it. (Scores low in absolute terms
because the synthetic siren is a pure harmonic stack, not a real siren
recording -- the RELATIVE comparison is the evidence.)

## 15. 2026-11-21 — Gap-closer outcomes (honest scoreboard)

- Multi-pitch (aligned hum): greedy 0/10 -> multi-pitch 2/10 (real 16 dB
  detections). Diagnosed: weak fundamental (+4.8 dB excess), envelope
  duty-cycling breaks persistence tests, broadband sources produce comb
  impostors. Needs a literature-grade harmonic-sieve estimator. OPEN.
- Multi-chunk narrowband (static-continuous +50 dB): diagnostic proved the
  weak source IS eigen-component 4 of the chunk-averaged covariance
  (corr 0.97 with true steering, +23 dB). Simple fixes (rank-4 MUSIC,
  loud-steering projection) failed because the near-field broadband
  interferer spans >=3 eigenvectors. Matched-subspace detection after
  blind loud-subspace removal remains the defined path. OPEN.
- Docs delivered: METRICS.md (plain-language metric glossary + retired
  metric warning) and PIPELINE.md (exact stage-by-stage spec, gain-norm
  formula and assumptions, how combinations are tested, no-ground-truth
  field reporting protocol).


## 16. 2026-11-21 — Full 13-scenario x 8-pipeline matrix; gain-norm trap quantified

- All five grid variants (LCMV/PhaseOnly/Robust/GSC+gn/DAS) fail target
  selection in ALL 11 interferer scenarios -- the algorithm-swap question
  is closed with data (INCREMENTAL_REPORT.md L1 verdict: NO).
- Tonality-argmax selection fails when an interferer is more tonal (S07)
  or the target is non-tonal (S12): detector-driven selection (S6) is
  load-bearing, not optional.
- L3b (MUSIC->SubspaceNull->comb): 16.7-16.9 dB under near-field bark +
  gain/phase errors; needs full-path extractor selection (S01 self-null).
- Gain-norm adversarial (user's concern, true 1/r): 2 m interferer bakes
  7 dB acoustics into 'gains', -1.3 dB when mis-applied cross-scene;
  per-chunk normalization = zero penalty. Rule: never transfer gains
  across scenes.
- Dual-LAM: AST 10-10 + AST 12-12 (balanced) as second opinion.

Dual-LAM cross-check: BLOCKED by network (12-12 variant weights download
stalls on SSL; script ready at experiments/lam_dual_check.py -- rerun when
the network allows, or with an HF_TOKEN for faster downloads). The AST
10-10 A/B (entry 14) already answers the LAM-coloration question.


## 17. 2026-11-21 — User challenge: normalization vs the history-anomaly
## detector; assumptions log; machinery QC

Pipeline model CORRECTED: company detection is per-beam POWER HISTORY +
anomaly + clustering (level CHANGE vs own history), not power-argmax; and
their LCMV peak-normalizes channels (max->1) per chunk. Consequence:
cross-chunk LEVEL STABILITY is the load-bearing quantity, and any
per-chunk data-driven normalization attacks it exactly when interference
changes.

Norm-ruin benchmark (60 chunks, constant-level target, interferer on in
chunks 20-39, +30 dB):
- fixed gains: target-beam onset shift = pure leakage only
  (+18.6 dB DAS / +7.3 dB LCMV)
- per-chunk RMS norm: onset shift -18.9 dB (DAS) / -30.2 dB (LCMV)
  -- the target beam's history COLLAPSES as an artifact
- company max->1 scheme: -21.7 / -29.6 dB + 3.5-4.5 dB target-gain jumps
  in delivered audio
- quiet scatter < 0.5 dB for ALL policies: the damage is
  interference-coupled, exactly as the user warned
NEW DESIGN RULE: normalization gains over long windows with hysteresis
(or coherence-weighted); level-history path uses unnormalized beams.

ASSUMPTIONS.md created (full assumption log with statuses). Benchmark
machinery QC'd (tests/test_qc_benchmarks.py, 5 checks incl.
Capon-map==sweep exactness and SI-SDR manual cross-check): all pass.


## 18. 2026-11-21 — Initial-changes benchmark (synthetic ESC-50 + REAL field
## data); user directive: field detectability is the final benchmark

USER DIRECTIVE: synthetic work is for reporting; the final benchmark for
every idea is DETECTABILITY ON THE FIELD DATA under the uncalibrated
array. Plan re-centered accordingly.

Initial small-changes menu (their architecture kept; both normalization
variants retained as requested), 6 REAL ESC-50 interferer classes:
- Synthetic @oracle: LCMV-family SI-SDR 4.9 dB / dSIR ~45 dB but
  LineSpread 3.2-3.5 dB; GSC 4.2 dB / 33.9 dB / 0.21 dB (flat).
- Stability (their anomaly statistic) with REAL interferers: plain
  LCMV/GSC onset +11.4/+12.0 dB (pure leakage); ALL per-chunk norm
  variants -14.9..-17.2 dB artifacts -- norm-ruin confirmed on real
  interferer classes.
- FIELD (4 persistent directions x 5 chunks, AST 10-10): every LCMV
  variant classifies "Vehicle" (0.27-0.37); GSC/DAS/PhaseOnly classify
  "Speech" (0.69-0.80) -- the LCMV coloration effect is REAL-DATA
  CONFIRMED on their recordings. Level stability: RMS->GSC 0.31-0.33 dB
  (best, 3x), MAX->GSC 0.79, plain GSC/DAS 0.95, LCMV ~1.05,
  PhaseOnly worst 1.1-1.3.

RECOMMENDED INITIAL CHANGE (minimal, their architecture): GSC + RMS
normalization for the extraction/classification path (class-correct
"Speech", flattest spectrum, phase-robust, best field level stability);
level-history path on UNNORMALIZED GSC (dual-path rule: per-chunk RMS
has the -15 dB onset artifact when interference patterns change).
Ambitious track unchanged: DOA-first shadow run, then per-candidate
tracks; BSS contingency.

## 19. 2026-11-21 — BENCHMARK_PLAN.md: unified scorecard framework

User request: named metric families per method/algorithm combination
(wind-proof, range of detection, detectability). Plan defines F1-F8
families (incl. R90 detection range and B-3dB robustness tolerance
points), C1-C10 condition axes, statistical protocol, the BENCH-v1
scorecard format, and the G1-G10 gap list with build order. Existing
evidence mapped; largest gaps: range envelope (G1), ROC harness (G2),
wind curve (G3).

## 20. 2026-11-21 — BENCHMARK_PLAN v2 after literature review

Reviewed: LOCATA (evaluation-on-data, motion-capture GT), DCASE SELD
(pre-registered 4-metric panel, dev/eval split), SiSEC/MDX (hidden test,
subjective), DOA Monte-Carlo practice (100-500 trials, RMSE-vs-SNR,
CRB), real-array datasets (dEchorate, RealMAN), wind literature (real
wind + coherence), stats practice (paired Wilcoxon, Holm, effect sizes),
task-driven evaluation trend. v2 adds: pre-registered factorial design
with power-based n (100/cell DOA, >=20 draws detection), ESC-50 fold
dev/test discipline, equalized tuning budgets, submission-style I/O,
paired stats harness (G13), CRB curves (G12), annotated field campaign
via ESC-50 playback at surveyed positions (G11). Existing synthetic
evidence downgraded to dev-split-grade until re-run under v2.

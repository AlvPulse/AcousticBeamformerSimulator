# Benchmark Plan: metric families x conditions x method combinations

Purpose: every method/algorithm combination gets ONE standardized scorecard.
Today's results are scattered across benchmark-specific tables; this plan
defines the unified metric families (including the requested "wind-proof"
and "range of detection" ratings), the condition axes, the statistical
protocol, what already exists, and what must be built.

---

## 1. Method combinations to score

Axis M1 beamformer: DAS | TaperedDAS | LCMV-SMI | RobustLCMV(CMT) |
PhaseOnly | GSC | SubspaceNull(k) | SOCP | EBB | MultiRankOSP | cGMM-MVDR |
SVE | INCM-MVDR | rake-MVDR (UNTESTED - near-field option)
Axis M2 normalization: none | per-chunk RMS | per-chunk MAX | long-window
(hysteresis) | coherence-weighted (to build)
Axis M3 localization: SRP-PHAT | NormMUSIC | narrowband@f0 | subspace-
removed | (FRIDA dropped)
Axis M4 pipeline level: L0 grid (company) | L1 swap | L3 DOA-first | L4 BSS
Axis M5 post-filter: none | oracle comb | tracked comb | multi-f0 (partial)

Full cross-product is not run; the plan scores the COMBINATIONS THAT
MATTER (current menu of ~20) on every applicable metric below.

## 2. Condition axes (the stress envelope)

| Axis | Sweep | Status |
|---|---|---|
| C1 target SNR at mic | -30..+10 dB | partial (fixed -10/-20 so far) |
| C2 interference INR | 0..+50 dB | DONE (points: 0/10/15/30/45/50) |
| C3 interferer type | tonal / broadband / intermittent / moving / continuous; REAL ESC-50 classes | DONE (6 classes + synthetic types) |
| C4 target range | 10 m..4 km (spreading + ISO absorption + near-field curvature) | NOT BUILT -> powers R90 metric |
| C5 wind | sigma_phi 0..0.4 (+ rain: NOT MODELED) | partial (0 / 0.12) |
| C6 gain miscal. | 0..12 dB spread | DONE |
| C7 phase miscal. | delay 0..4 cm; all-pass 0..45 deg | DONE |
| C8 simultaneous sources | 1..4 | partial (2..3) |
| C9 dead/degraded mics | 0..8 mics dropped | NOT BUILT |
| C10 reverberation | RT60 0..0.8 s | NOT MODELED (outdoor deferred) |

## 3. Metric families (the scorecard rows)

### F1 Detection metrics (needs ground truth = synthetic; field via their detector)
- **P_d** @ fixed P_fa: probability the pipeline reports the target
  (selection correct AND tonality/AST above threshold). Protocol: 20+
  draws per condition cell, threshold frozen on a dev set.
- **FA/hour**: false candidate reports per hour of interferer-only audio.
- **d-prime**: separability index between target-present and target-absent
  score distributions.
- **R90 -- RANGE OF DETECTION** (user-requested): max target range with
  P_d >= 0.9 at nominal noise/interference. Protocol: sweep C4 range
  (10 m..4 km) with ISO 9613 absorption (gen-1 code has it); report R90
  per method. TO BUILD (script: bench_range_envelope).

### F2 Localization metrics
- DOA error median + P90 (deg); source-count accuracy; spurious rate;
- angular resolution: minimum separation resolving two sources (sweep
  2..30 deg). TO BUILD.
- near-field bias: DOA error vs range < 50 m (rake-MVDR candidate axis).

### F3 Separation metrics (synthetic only)
- SI-SDR, purity/dSIR, SIR improvement. DONE (METRICS.md definitions).

### F4 Fidelity metrics (downstream-classifier impact)
- line-gain spread (dB) DONE; LSD DONE;
- **AST class-preservation score**: fraction of extractions where AST's
  top class equals the clean-reference top class (machinery exists;
  formalize as metric + threshold). PARTIAL -> formalize;
- AST confidence delta vs clean reference.

### F5 Robustness ratings (user-requested "wind-proof" style)
Every F1-F4 metric is also measured ALONG a condition axis; the rating is:
- **Degradation curve**: metric vs axis level (gain dB, phase deg, wind
  sigma, INR dB);
- **B-3dB point**: the axis level where the metric drops 3 dB below its
  calibrated value = the method's tolerance rating. E.g. "phase-proof:
  12 deg", "wind-proof: sigma 0.25", "gain-proof: 9 dB".
- Wind rating additionally at two ESC-50 wind clips (real) + synthetic
  decoherence. PARTIAL (phase/gain curves DONE -> extract B-3dB points;
  wind curve TO BUILD).

### F6 Stability metrics (their anomaly detector interface)
- onset artifact (dB), quiet scatter (dB), false-anomaly count under the
  norm-ruin protocol. DONE (both norms kept).

### F7 Cost metrics
- RTF, peak RAM, latency. DONE (RTF/RAM; single-machine caveat).

### F8 Field metrics (no ground truth -- final benchmark per user)
- cross-estimator agreement (deg); persistence (chunks survived);
  tonality & AST-confidence improvement over raw mic; track coherence;
- FINAL: their detector's judgments on frozen candidate sets = the only
  reported "detection" numbers on real data. Protocol in PIPELINE.md.

## 4. Statistical protocol (all cells)
>= 20 draws/cell for F1/P_d (Wilson 95% CI); paired scenes across methods;
frozen seed list; dev/test split for thresholds; every table reports n.

## 5. Gap list -> build order

| # | Item | Feeds | Effort |
|---|---|---|---|
| G1 | Range sweep w/ absorption (C4) + R90 | F1 | 0.5 d |
| G2 | Detection/ROC harness (score dists, thresholds, P_d/FA/d') | F1 | 1 d |
| G3 | Wind curve + B-3dB extraction for all F5 ratings | F5 | 0.5 d |
| G4 | AST class-preservation metric (F4) formalized | F4 | 0.2 d |
| G5 | Dead-mic axis C9 | F5 | 0.3 d |
| G6 | Angular resolution sweep | F2 | 0.3 d |
| G7 | Long-window + coherence-weighted normalizers (M2) | F5/F6 | 0.5 d |
| G8 | rake-MVDR near-field option (M1) | F2/F3 | 0.5 d |
| G9 | Rain + reverb physics (C5/C10) | F5 | deferred w/ user |
| G10 | Field detector-in-loop harness (their thresholds) | F8 | needs their side |

## 6. Output format (BENCH-v1 scorecard)

One row per method combination; columns grouped by family:
`method | P_d | FA/h | R90_m | DOAerr_p90 | SI-SDR | dSIR | linespread |
ASTpres | gain-proof_dB | phase-proof_deg | wind-proof_sigma | onset_artifact_dB |
level_stab_dB | RTF`
Every cell: value [CI] @ reference conditions; degradation curves as
separate annex sheets. Stored in docs/bench_v1/ as CSV + generated report.

## 7. What already exists (mapped)

mc_benchmark.csv -> F1(partial)/F5(gain) for B-family; phase_robustness.csv
-> F5(phase) full curves; industrial_benchmark.csv -> F5(INR-type);
norm_ruin + initial_changes stability -> F6; all_pipelines_benchmark.csv ->
F1-F3 across L0-L4; field_quick_pass + field_candidates + initial_changes_
field -> F8 partial. Missing cells are exactly G1-G8.

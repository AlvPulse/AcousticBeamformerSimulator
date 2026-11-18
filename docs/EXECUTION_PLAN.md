# Execution Plan: From Fixed Simulator to Company Decision

**Goal:** Validate Pipeline B (DOA-first: MUSIC/SRP → per-candidate robust extraction →
feature-based target ID) on the company's own field data, and package the evidence so
the company can decide to move from grid-scan LCMV (Pipeline A) to B — with A retained
as reference and C (blind separation) as a measured contingency.

**Where we stand (2026-11):** steering-conjugation bug fixed and regression-tested
(12/12); all 15 beamformers re-benchmarked under clean/6 dB/12 dB gain imbalance;
pipeline-system benchmark shows A and A' (algorithm swap) fail target selection
(~100 deg DOA error, every seed), B is the strongest system, GSC-MCWH is the best
imbalance-safe extractor, all single-output gridless methods return the loudest
source. Field data (652 chunks, 64 ch, 8 kHz) and the 5-phase field plan exist but
have NOT been run.

---

## Phase 0 — Repo and evidence hygiene (half day)

1. Commit all fixes on `fix/steering-and-benchmarks` (base.py, scene_generator,
   robust_lcmv, robust_phase, tests, benchmark scripts, result CSVs); push; open PR
   against the working branch.
2. Remote-branch decision: cherry-pick CODE from `origin/robust-sota-*` only
   (music/srp_phat/incm_mvdr, run_full_pipeline.py). Do NOT merge its docs/.docx —
   every conclusion table predates the steering fix and is invalid; regenerate
   reports from re-run data instead.
3. `requirements.txt`: add cvxpy, tabulate, pyroomacoustics (pinned versions from .venv).
4. Housekeeping: gitignore `benchmarks_*.log`; commit `array_geo.csv` (field pipeline
   needs it); decide fate of `experiments/test.ipynb` (commit trimmed or delete).
5. Robustness guard: LCMV pinv-stall outlier (seed-2 clean, RTF 7813) — enforce a
   covariance-loading floor and skip pathological bins.

**Gate 0:** clean tree, 12/12 tests, PR open, requirements reproducible.

## Phase 1 — Make the synthetic evidence defensible (1–2 days)

1. Monte Carlo: 20 draws x 3 gain conditions for systems A, A', A'', B, B'(x2
   extractors), C. Report selection reliability with confidence intervals — not means.
2. B' v2: NormMUSIC -> GSC-MCWF extraction (GSC beat PhaseOnly at 6/12 dB).
3. Scene realism: add reverberation/ground reflection (pyroomacoustics Room,
   image-source) and wind phase decoherence; use real drone audio from
   `data/real_world inputs` as source material where usable.
4. Near-field variant: SRP mode='near' + rake-MVDR for sources < ~15 m (2D^2/lambda).
5. Fix the selection metric: recalibrate harmonic score with noise-floor
   normalization (it currently prefers noisy outputs — scored clean target -74 dB),
   or replace with a spectral-coherence score; SI-SDR stays synthetic-only.
6. FRIDA multi-source on a synthetic 64-mic geometry (works there, fails at M=8).

**Gate 1:** frozen results tables v2 with CIs; no metric pathologies; B selection
reliability known (needed: >= 90% on synthetic to pitch).

## Phase 2 — Field-data pass: the decisive evidence (1–2 days)

Scripts: `experiments/field_qc.py`, `experiments/field_doa_tracks.py` (new).

- **2.0 Hardware QC:** channel health (dead/quiet/clipped), inter-channel sync lags,
  measured per-mic gain spread (answers "is the hardware really 6 dB imbalanced?"),
  channel-to-geometry map verification via correlation swaps.
- **2.1 DOA pass (all 652 chunks):** SRP + NormMUSIC + FRIDA, 250–1200 Hz band
  (array aliases above ~1.2 kHz), K=3–4 candidates, cross-estimator agreement flags.
  Planar-array mirror ambiguity: sky-hemisphere assumption, elevation soft.
- **2.2 Tracking:** gated nearest-neighbor association across the 2.12 s chunks ->
  per-candidate time series (DOA, energy, harmonic features). The ten 2024 sessions
  (60 chunks each) are natural repeats for coherence checks.
- **2.3 Extraction at candidates only:** GSC + PhaseOnly, <= 4 WAVs/chunk; harmonic
  + optional LAM features per candidate. No artificial gain injection.
- **2.4 BSS cross-check:** OverIVA on ~10% sample + all estimator-disagreement
  chunks; agreement between geometry-based and separation-based DOA = confidence.

Deliverables: `candidates.csv`, `track_summary.csv`, per-candidate WAVs, review
heatmaps, findings memo.

**Gate 2 (go/no-go for pitch):** tracked candidates form physically plausible,
temporally coherent objects; estimator agreement acceptable. If DOA is garbage on
real data, the pitch becomes a calibration-audit proposal instead — surface this,
don't hide it.

## Phase 3 — Company decision package (1 day)

1. The decisive number: run THEIR detector/classifier on identical chunk sets from
   (a) their current pipeline outputs and (b) our B outputs; frozen thresholds;
   report precision / recall / false-alarms-per-hour and compute cost.
2. Pitch structure: A's measured selection failure -> A' (swap) result -> why B
   (same interface as their history/clustering, per-candidate time series) ->
   imbalance & cost tables -> risks honestly -> decision gates.
3. Adoption proposal: shadow mode — run B in parallel with A for 2 weeks on live
   data, compare detections, then decide cutover. Lowest-risk path for a PM to say yes.

## Phase 4 — Conditional productionization (post-pitch, scoped after Gate 3)

- Harden field scripts into `evaluate_field_v2.py` (streaming/blockwise, no 19x64 scans).
- Embedded RTF budget on target hardware; tracking upgrade (gating/JPDA); threshold
  tuning protocol with frozen test sets; monitoring & drift alerts.
- OverIVA contingency track only if calibration drift proves worse than Phase 2.0
  measurements.

---

## Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| Field DOA unreliable | Gate 2 fails | Calibration audit becomes the pitch; PHAT robustness already measured |
| Feature-ID selection unreliable | MC reliability < 90% | Use their detector directly per candidate (Phase 3 design) |
| Synthetic-to-real gap | Phase 2 disagrees with Phase 1 | Realism items 1.3–1.4; report deltas openly |
| "We already invested in A" | Organizational | Shadow-mode framing; A stays reference throughout |
| BSS variance | C results erratic | C is contingency, never primary |
| Elevation ambiguity | Planar array mirror | Sky-hemisphere assumption + elevation as soft feature |

## Explicitly not doing

- Merging bug-contaminated remote docs/reports (regenerate instead).
- Pitching C (blind separation) as the primary system.
- Promising pure extraction (measured best: 0.5–1% interference residual clean).
- Rewriting their downstream detection/clustering — B feeds it per-candidate.

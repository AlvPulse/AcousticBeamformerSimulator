# Benchmark Plan v2 — scientifically & statistically verifiable

Revision after literature review (sources in §8). v1's metric families and
condition axes are kept; this revision adds the experimental-design,
fair-comparison, and statistical-verification rules the literature treats
as standard, and the field-test showcase protocol.

---

## 0. What the literature does (and we adopt)

1. **LOCATA (IEEE AASP challenge)**: open recordings + motion-capture
   ground truth; fixed task list (static/moving sources & arrays,
   ego-noise); *evaluation-on-data* — every algorithm consumes identical
   input files and emits a fixed output format; angular/distance error
   and tracking measures. -> We adopt submission-style harness identical
   inputs/outputs per method, and recommend an annotated field campaign
   (Gap G11).
2. **DCASE SELD**: a fixed 4-metric panel (ER/F1 detection + LE/CD
   localization-error/recall, class-dependent), public dev/eval split,
   official baselines, ranking only by pre-declared metrics. -> We adopt
   a pre-registered metric panel (F1-F8) and dev/eval splits; no metric
   shopping.
3. **SiSEC/MDX**: hidden test set for fairness; standardized objective
   metrics + large-scale subjective listening to catch what SDR misses.
   -> We adopt held-out test scenes (never used for tuning) and flag
   listening tests as optional G15.
4. **DOA Monte-Carlo literature**: RMSE-vs-SNR and RMSE-vs-snapshots
   curves with 100-500 trials, fixed source angles, and the Cramer-Rao
   bound as the scientific reference. -> We adopt 100-trial RMSE curves
   for DOA (not just hit-rate) and CRB reference curves (G12).
5. **Real-array datasets (dEchorate, RealMAN, MIRaGe)**: benchmarking on
   real recordings WITH annotations (RIRs, positions). -> Reinforces G11:
   our field campaign should include a few annotated takes.
6. **Wind literature (Grimm 2018; Lyons JASA)**: turbulence shorter than
   the aperture is INCOHERENT across mics -> wind evaluation must include
   real wind recordings + inter-mic coherence statistics, not only STFT
   phase-jitter synthesis. -> Wind axis gets real-wind condition (G3
   extended).
7. **Statistical practice (Dror et al. 2018; Urbano et al.)**: paired
   tests on shared items, multiple-comparison correction, effect sizes
   beyond p-values, report n everywhere. -> Adopted verbatim (§3).
8. **Task-driven evaluation trend (MERL TUSS; Separate-and-Detect;
   LibriCSS)**: judge separation by the downstream task (WER, detection
   AUROC), not SI-SDR alone. -> Our downstream metric = AST
   class-preservation + their detector (F4/F8), now first-class.

## 1. Experimental design (pre-registered)

**Factorial grid** (full factorial on the axes that matter; blocked
everywhere else):
- F-beam: {DAS, LCMV, RobustLCMV, PhaseOnly, GSC, SubspaceNull} (+rake in G8)
- F-norm: {none, RMS, MAX, long-window}
- F-level: {L0, L1, L3, L4}
- C-SNR: {-30,-20,-10,0} dB | C-INR: {0,15,30,45,50} dB
- C-interferer: ESC-50 classes stratified across predefined FOLDS:
  dev = folds 1-3, test = folds 4-5 (clip-disjoint; ESC-50's own fold
  structure, so no class or clip leaks)
- C-geometry: target az uniform; interferer az with >=30 deg separation;
  N-interferers {1,2,3}
- C-range: {10,50,200,1000,4000} m (G1) | C-wind: {none, synth sigma 0.12,
  0.25, real-wind clips} | C-cal: {gain 0/6/12, phase 0/20/45}

**Sample sizes (power-based, pre-registered):**
- DOA RMSE curves: **100 trials/cell** (matches DOA literature; detects
  0.5 deg RMSE differences)
- Detection P_d / scorecard cells: **>= 20 draws** (paired; for paired
  detection of a 1.0 dB SI-SDR difference at sigma_d=0.8 dB, alpha=0.05,
  power 0.8 -> n=~10; 20 gives margin) with Wilson CIs
- Stability/anomaly: 16-chunk protocol fixed (as norm-ruin), 5+ scenes
- Seeds: frozen seed-list file committed BEFORE runs; every table states n.

**Fair-comparison rules (non-negotiable):**
1. Paired scenes: every method runs on the SAME draw (same seeds).
2. Tuning budget equalized: every tunable method (SOCP eps, CMT width,
   SubspaceNull k, comb width...) is tuned ONLY on dev scenes with a
   fixed budget (grid of <= 5 values), then FROZEN for test.
3. No oracle information in headline numbers; oracle-informed variants
   (oracle f0, oracle extractor choice) are labeled 'oracle ablation'.
4. Identical I/O: submission-style harness -- each method reads the same
   chunk files, writes the same output schema (LOCATA practice).
5. Compute reported next to every accuracy number (RTF, RAM).

## 2. Metric panel (pre-registered; no post-hoc additions)

v1 families F1-F8 stand, with these upgrades:
- F1 gains DCASE-style **LE(deg) / LR** localization error/recall in
  addition to hit-rate, and **d-prime** from score distributions.
- F2 gains **RMSE-vs-SNR and RMSE-vs-snapshot curves (100 trials)** and
  **CRB reference** (G12) -- 'how far from optimal is this method' is a
  scientifically verifiable statement.
- F4 (downstream) becomes co-primary with F1: AST class-preservation
  rate + AST-confidence delta (task-driven evaluation per literature).
- F5 B-3dB robustness ratings computed from fitted degradation curves
  (monotone spline fit; rating = curve crossing; CI via bootstrap).
- F8 field metrics unchanged + leave-one-session-out threshold transfer
  (see §4).

## 3. Statistical verification protocol

- All method comparisons are **paired** (same scenes): paired Wilcoxon
  signed-rank (n>=20) or paired t where normality holds (Shapiro check);
  report effect size (Cliff's delta) with every p-value.
- **Multiple comparisons**: Holm-Bonferroni across the method family
  tested within one table.
- CIs: Wilson (rates), bootstrap BCa (degradation curves, R90).
- Pre-registration: seed list + factor grid + metric panel committed
  before running the test split; deviations logged in DECISION_LOG.
- Repeatability: pinned versions (requirements.txt), frozen seeds,
  one-command rerun per benchmark.

## 4. Field-test showcase (the final benchmark, per user directive)

Synthetic scorecard -> field, in the LOCATA/DCASE spirit:
1. **Dev/eval split by session**: thresholds (tonality, AST confidence,
   anomaly z) tuned on the ten 2024 sessions; the 2026 field_tests
   session + any new recordings = held-out eval. Leave-one-session-out
   cross-validation for stability of thresholds.
2. Full 652-chunk run, submission-style: every method emits the same
   candidate table; metrics = F8 panel (agreement, persistence,
   tonality/AST improvement over raw mic, track coherence).
3. **Detector-in-loop**: their classifier + their thresholds score frozen
   candidate sets from each method -- the only reported 'detection'
   numbers on real data.
4. **Annotated field campaign (G11, recommended)**: a few takes with
   known source positions/times (GPS, laser, or playback of known clips
   from known positions -- ESC-50 clips replayed at measured ranges!).
   This converts the field benchmark from consistency-based to
   ground-truth-based, matching LOCATA/RealMAN practice. Playback of
   ESC-50 clips at surveyed positions is cheap and immediately doubles
   as real-interferer validation.

## 5. Revised gap list (v2)

| # | Item | Feeds | Status |
|---|---|---|---|
| G1 | Range sweep + R90 | F1 | planned |
| G2 | ROC/Pd harness + d-prime | F1 | planned |
| G3 | Wind axis: synthetic sigma sweep + REAL wind clips + coherence stats | F5 | extended per lit |
| G4 | AST class-preservation metric | F4 | planned |
| G5 | Dead-mic axis | F5 | planned |
| G6 | Angular resolution sweep | F2 | planned |
| G7 | Long-window + coherence-weighted norms | F5/F6 | planned |
| G8 | rake-MVDR | F2/F3 | planned |
| G9 | Rain/reverb physics | C5/C10 | deferred (user) |
| G10 | Their detector-in-loop harness | F8 | needs their side |
| **G11** | **Annotated field campaign (playback at surveyed positions)** | F8->GT | **new, recommended** |
| **G12** | CRB reference curves for DOA | F2 | **new** |
| **G13** | Paired-stats harness (Wilcoxon+Holm+Cliff's delta) | all | **new** |
| **G14** | ESC-50 fold discipline (dev 1-3 / test 4-5) | design | **new** |
| G15 | Optional listening test on field candidates | MDX practice | optional |

## 6. Execution order

G13+G14 first (harness + discipline), then G1+G2+G12 (detection/R90/CRB),
G3+G4 (wind + downstream), then the full factorial on dev, freeze, and
the test-split run; field showcase per §4 in parallel with G11 campaign
planning.

## 7. Existing evidence -> v2 mapping

mc_benchmark, phase_robustness, industrial, norm_ruin, initial_changes,
all_pipelines (synthetic) = dev-split-grade evidence (re-run under v2
rules where n<20 or tuning leaked); field_quick_pass, field_candidates,
initial_changes_field = F8 partials. Nothing synthetic is cited as final
until re-run under the pre-registered protocol.

## 8. Sources

- LOCATA: arxiv.org/abs/1909.01008; lms.tfau.eu/locata-datasets
- DCASE SELD: dcase.community/challenge2023/...real-spatial-sound-scenes;
  2019/2020 task reports
- SiSEC/MDX: arxiv.org/abs/2108.13559 (MDX21); TISMIR SDX'23
- DOA Monte-Carlo practice: Khmou 2014; Muni 2026 (Sci. Rep.);
  Zhao 2025 (500-trial MC)
- Real-array datasets: dEchorate (arxiv 2106.06859); RealMAN
  (arxiv 2406.19959); MIRaGe
- Wind: Grimm et al. 2018 (EURASIP JASMP); Lyons et al. (JASA)
- Statistics: Dror et al. ACL 2018; Urbano et al. (effect sizes)
- Task-driven eval: MERL TUSS (arxiv 2609.11092); Separate-and-Detect
  (arxiv 2608.01093); LibriCSS

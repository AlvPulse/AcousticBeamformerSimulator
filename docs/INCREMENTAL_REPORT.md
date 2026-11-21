# Incremental Buildup Report: benchmarking the company gold standard

**Goal restated.** The company standard is: grid LCMV scan -> per-node
history anomaly detection -> clustering -> classification/detection.
We benchmark it and answer, with evidence: what is the minimum change that
fixes each problem, and what is the right endpoint?

**Problems identified (all measured, see DECISION_LOG.md):**
P1 target selection: energy-argmax lands on the loudest interferer
   (0% correct selection in 60+ randomized draws) -- architectural.
P2 uncalibrated array: gain (8.1 dB measured on our hardware) and phase
   errors destroy LCMV-family extraction (9.8 -> 0.1 dB SI-SDR at +/-20 deg).
P3 spectral fidelity: LCMV's per-bin gains color the target audio
   (2.5-3 dB line-gain spread calibrated; 8-10 dB under phase error) and
   change what the classifier hears (AST: "Siren" -> "Wind instrument"/
   "Waterfall").
P4 extreme interference: close interferers at +40..50 dB over the target.

---

## Level 1 -- keep the pipeline, swap the algorithm. Is it enough?

| Swap | P1 selection | P2 uncalibrated | P3 fidelity | P4 extreme | Verdict |
|---|---|---|---|---|---|
| PhaseOnly LCMV | 0% (unchanged) | better map, extraction 4-8 dB | moderate coloration | poor | NO |
| RobustLCMV (real CMT) | 0% | no better than LCMV under phase err | as LCMV | partial | NO |
| GSC + gain-norm | 0% (unchanged) | 10.2 dB @ +/-6dB gain, 12.2 dB @ +/-20 deg phase | flat (0.2-1.2 dB spread), AST keeps "Siren" | leaks at +40..50 dB | BEST SWAP, still NO |

**Conclusion L1: no algorithm swap fixes P1.** Selection-by-energy fails
regardless of the beamformer inside the scan; even grid + auto-calibration
(A+) stays at 0%. A swap (GSC+gain-norm) is however a strict upgrade for
P2/P3 if the architecture stays -- a valid *interim* step requiring zero
pipeline change.

## Level 2 -- change the spatial method entirely (DAS / MVDR / rake)

- DAS: immune to calibration (fixed weights), flat spectrum, but no
  interference rejection (SI-SDR ~4 dB everywhere) and selection still 0%.
- MVDR/LCMV = already the standard; rake/regularized variants (pyroomacoustics
  `rake_mvdr`) are near-field-aware upgrades -- untested here, listed as
  the one literature option not yet benchmarked (near-field sources).
- Verdict: changing the spatial method alone still leaves P1/P4.

## Level 3 -- change the architecture: DOA-first (B)

NormMUSIC/SRP candidates -> per-candidate extraction -> spectral
decomposition -> track persistence -> detector.
- P1 SOLVED: 100% selection in 90 randomized draws (MC benchmark).
- P2: GSC+gain-norm extractor -> 10.2 dB under 12 dB gain spread.
- P3: GSC keeps the audio classifier-recognizable.
- P4: +36-42 dB cascade rejection (spatial+comb) when separated;
  loud sources (bark/urban/movers) 100% blind in the 30-draw blind-MC.
  Weak target under STATIC-CONTINUOUS +50 dB remains open per-chunk
  (multi-chunk integration path defined, DECISION_LOG #15).
- Cost: RTF ~0.15 vs 1.5-3.4 for grid scans (10-20x cheaper).

## Level 4 -- BSS-first (C) and hybrid BSS+beamforming

- OverIVA multi-output + manifold DOA: works, gain-invariant (identical
  SI-SDR at 0/6/12 dB), but high variance and weaker extraction (6.4 dB);
  all single-output blind extractors return the loudest source.
- Hybrid = exactly our S2+S3 (beamformer first for directionality, spectral
  separation after) -- the measured +16-21 dB comb contribution on top of
  any spatial stage is the hybrid's value.

## Literature options map (what we tested vs not)

| Option | Status |
|---|---|
| Grid scan (company) | benchmarked (L0) |
| Algorithm swaps (PhaseOnly/CMT/GSC) | benchmarked (L1a-c) |
| SRP/MUSIC DOA-first | benchmarked (L3) |
| Subspace nulling / EBB-style | benchmarked (SubspaceNull) |
| Overdetermined BSS (OverIVA) | benchmarked (L4) |
| Hybrid BF+spectral (comb/NMF) | comb benchmarked; NMF/learned mask open |
| Rake/regularized BF (near-field) | NOT benchmarked -- next |
| Multi-pitch estimation | partial (0->2/10), open |
| Multi-chunk coherent integration | mechanism proven (eig4, corr 0.97), open |
| Learned end-to-end (neural beamforming) | out of scope for now |

## Scenario matrix (13 scenarios x 8 pipelines) -- INSERTED AFTER RUN

See docs/all_pipelines_benchmark.csv. Key failure map:
| Scenario | Grid family (L0/L1a/L1b/L1c/L2) | L3a MUSIC->GSC+gn | L3b MUSIC->SubspaceNull+comb |
|---|---|---|---|
| S01 quiet (sanity) | OK | OK (15.7 dB) | FAIL (fixed extractor self-nulls the only source -> extractor selection is mandatory, not optional) |
| S02 interferer +10 dB | 0% sel, ALL 5 | OK 13.4 dB | OK (0.1) |
| S03 far interferer +45 dB | 0% sel | 46 deg err | 100 deg err |
| S04 NEAR-FIELD bark +45 | 0% sel | OK sel, 1.0 dB | **OK 16.9 dB** |
| S05 continuous machine +50 | 0% sel | 84 deg err | partial (36 deg, 0.8 dB) |
| S06 moving car +45 | 0% sel | sel OK, 0.0 dB | sel OK, 0.3 dB |
| S07 tonal hum louder+more tonal | 0% sel | 80 deg err (selector picks the MORE TONAL interferer!) | **OK** |
| S08 two weak tonals | OK | OK 12.9 | OK sel |
| S09 = S04 + 12 dB gain imb. | 0% sel | OK sel, 1.0 dB | **OK 16.7 dB** |
| S10 = S04 +/-20 deg phase | 0% sel | OK sel, 1.0 dB | **OK 16.8 dB** |
| S11 = S05 + wind | 0% sel | 86 deg err | partial (sel 2 deg, 3.3 dB) |
| S12 CHIRP target | 0% sel | 12 deg (selector cannot rank chirps) | fail |
| S13 machinery target | 0% sel | **OK 17.9 dB** | OK sel |

Three structural conclusions:
1. **All five grid variants fail selection in all 11 interferer scenarios**
   (80-120 deg errors) -- algorithm swaps do not touch P1. Definitive.
2. **Selection-by-tonality fails when an interferer is MORE tonal than the
   target (S07) or the target is non-tonal (S12 chirp)** -> the detector
   (LAM) as the selection stage is REQUIRED, not an optimization. This
   upgrades S6 from 'best practice' to 'load-bearing component'.
3. L3b (subspace+comb) is the strongest extractor under combined stress
   (16.7-16.9 dB at near-field bark + gain/phase errors) but needs the
   full-path extractor selection to avoid its S01 self-nulling failure.

## Gain normalization -- the trap, measured

The user's concern: estimating 'gains' on a scene dominated by near-field
interferers bakes ACOUSTIC level differences into 'hardware' gains, then
mis-applies them to far-field data.
Measured with true 1/r amplitudes: a 2 m interferer bakes 7.0 dB of
ACOUSTIC level variation into an RMS 'gain' estimate; applying those
gains to a far-field scene costs 13.03 -> 11.73 dB SI-SDR (-1.3 dB).
At 5 m the penalty is -0.2 dB. Per-chunk normalization on the SAME chunk
it is applied to: **zero penalty** in all cases (13.03 dB throughout).

Rule adopted: gain-norm is computed per-chunk on the SAME chunk it is
applied to (never transferred across scenes), and only as the GSC
front-end normalization; a coherence-weighted estimator is specified for
the field version.

## Recommended incremental path to the company

1. Interim (their pipeline unchanged): swap LCMV -> GSC + per-chunk
   gain-norm. Fixes P2/P3 immediately; zero architecture change.
2. Parallel shadow: run L3 (DOA-first + persistence) beside their system;
   their detector judges identical candidate WAVs (Phase-3 protocol).
3. Cut over node-level anomaly logic to per-candidate track features.
4. Keep L4/BSS as contingency for calibration-hostile installs.
5. Open items before full field deployment: multi-pitch, multi-chunk
   integration, rake-MVDR for near-field sources.

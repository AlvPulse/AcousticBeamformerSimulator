# Assumptions Log (every assumption this project makes, and its status)

A = assumed & unvalidated | T = tested/validated (evidence) | C = CORRECTED
after user input | O = open

## Company pipeline model
- [C] Detection statistic: initially assumed power-argmax over grid nodes.
  CORRECTED (user, 2026-11-21): anomaly detection on each beam's POWER
  HISTORY + clustering; the statistic is level CHANGE relative to the
  beam's own history. Consequence: cross-chunk level stability is the
  sacred quantity (tested in norm-ruin benchmark).
- [C] Their LCMV peak-normalizes channels (max -> 1) per chunk before
  filtering (user report). Consequence modeled and measured: the
  normalizing constant is set by the loudest content, so interferer
  onsets move every beam's level by -22..-30 dB artifactually.
- [A] Their downstream: node history features -> anomaly -> clusters ->
  classification/detection; thresholds tuned on their data (not ours).

## Scenes / synthesis
- [T] Point sources, free field; exact fractional delays (validated vs
  pyroomacoustics DOA on same data).
- [A] No reverberation or ground bounce (wind = STFT phase decoherence
  only); no rain model.
- [T] Source classes: tonal stacks (350/700/1050), chirps 1.8-3.5 kHz,
  bark-like bursts (300-3000 Hz, ~24% duty), urban hum (430/860 + band),
  machinery (220/440 + harsh noise). All synthetic; NO real drone/bird
  recording used yet (field 'real_world inputs' are mixtures, not sources).
- [A] Envelopes: slow sinusoidal AM (0.6-1.9 Hz); bark burst times random.
- [A] Sensor noise white, target at -10..-20 dB over noise per mic.

## Array
- [A] array_geo.csv coordinates are TRUTH; channel i in wav == column i
  (cross-check pending in field QC).
- [T] Planar z=0 -> (az,el) vs (az+180,-el) ambiguity; sky-only assumed;
  elevation treated as soft information.
- [T] Alias-free band 250-1100 Hz for DOA (spacing ~0.13 m).
- [A] Single clock, synchronous channels (sync check pending).

## Calibration / channel models
- [T] Gain errors modeled as scalar per mic (measured level spread on
  field data: 8.1 dB = upper bound incl. acoustics).
- [T] Phase errors modeled as per-mic delay (+/-0.5-4 cm) or all-pass
  (+/-10-45 deg); NO frequency-dependent channel response modeled.
- [O] Coherence-weighted gain estimator specified, not yet benchmarked.

## Normalization policies (norm-ruin benchmark)
- [T] Per-chunk RMS norm: zero waveform SI-SDR penalty within a chunk;
  but the target-beam LEVEL SERIES collapses -19..-30 dB at interferer
  onset (artifact) -- destructive to history-anomaly detection.
- [T] Per-chunk max->1 (company scheme): -22..-30 dB onset artifact +
  3.5-4.5 dB target-gain jumps in delivered audio.
- [T] Fixed gains: onset shift = pure leakage only (+7.3 dB LCMV,
  +18.6 dB DAS).
- [T] Quiet scenes: all policies stable (<0.5 dB scatter) -- the damage
  is interference-coupled, not stationary noise.
- NEW DESIGN RULE (from this test): normalization gains must be estimated
  over LONG windows with hysteresis (or coherence-weighted), never
  per-chunk; the level-history path must use unnormalized beams.

## Benchmark machinery (QC'd in tests/test_qc_benchmarks.py)
- [T] Capon-map == grid-argmax of LCMV scan (exactness claim QC-tested).
- [T] SI-SDR implementation sanity (scale invariance, manual cross-check).
- [T] Tonality selector direction on known tonal vs broadband inputs.
- [T] Gain round-trip: RMS estimator recovers known gains in far-field
  scenes within 0.5 dB.
- [T] Per-chunk gain stability in quiet scenes (<0.1 dB).
- [A] RTFs are single-machine (this PC), not embedded targets.

## Statistics
- [T] 10-20 draws with Wilson CIs where claimed; single-draw tables are
  labeled as such (extreme-INR, method matrix, pipeline matrix rows).
- [A] Draw independence; seed-reproducible.

## Field evidence
- [T] 52 of 652 chunks QC'd (levels, dead channels, MUSIC-vs-SRP 1.2 deg
  agreement, 4 persistent directions). No ground truth exists; reporting
  follows the reference-free protocol (PIPELINE.md).

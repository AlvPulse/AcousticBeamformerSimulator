# LCMV Replacement: Theoretical Comparison (why, in engineer terms)

The company grid pipeline replaces only the per-beam beamformer. This note
explains, per candidate, WHY it behaves the way our benchmarks measure —
tied to the math, with the measured numbers referenced. Companion
synthetic evidence: phase_robustness / mc / initial_changes / industrial
CSVs. Real-world evidence: docs/field_stft/ + field_stft_analysis.csv.

---

## 1. LCMV-SMI (their baseline) — where it breaks

Per frequency bin: w(f) = R(f)^-1 a(f) / (a(f)^H R(f)^-1 a(f)).

- **The distortionless constraint holds only at the NOMINAL steering
  a(f).** With any channel mismatch (gain, phase, position error), the
  true arrival a_true != a, so the gain |w^H a_true| varies per bin —
  measured: 2.5-3 dB line-gain spread calibrated, 8-10 dB at +-2 cm
  position error; AST then hears "Vehicle" where DAS/GSC hear "Speech"
  (p_holm = 1.2e-9 on their own recordings).
- **Nulls are covariance-driven:** with miscalibration, R is built from
  the mismatched data, so nulls point at wrong angles — measured: 9.8 dB
  SI-SDR calibrated -> 0.1 dB at +-45 deg phase error (phase_robustness).
- Amplitude fidelity: the constraint fixes gain only at one direction;
  everything else rides on R(f), which changes per bin and per chunk.

## 2. PhaseOnly LCMV (their candidate of interest)

Normalizes each STFT bin to unit magnitude: X -> X/|X|; covariance from
phase-only vectors.

- **Why it is gain-invariant:** a per-mic gain g_m multiplies both X_m
  and |X_m|, so X/|X| is unchanged. Amplitude calibration becomes
  irrelevant — a real, provable property, and the right instinct for an
  uncalibrated array.
- **The cost (also provable):** null depth comes from the
  interference-to-noise EIGENVALUE CONTRAST in R. Unit-magnitude
  normalization flattens that contrast (every mic contributes equal
  power per bin), so R^-1 stops allocating deep cancellation — measured
  ~12.7 dB shallower SSR than LCMV in clean scenes; and its
  Ledoit-Wolf shrinkage (mean 0.72 in our instrumented runs) pushes
  weights toward plain DAS.
- **Spectral fidelity:** still a per-bin adaptive w(f) applied to the
  ORIGINAL (un-normalized) signal — so it inherits LCMV's per-bin
  coloration (line spread 3.9 dB, the worst measured) while losing
  null depth. On their field data it classifies "Speech" (unlike LCMV)
  but with the least stable level series (1.1-1.3 dB).
- Verdict: solves the GAIN axis only; phase errors still hurt (it is
  phase-blind by design, so per-bin PHASE mismatch still misplaces
  nulls); gives up rejection and keeps the coloration.

## 3. GSC + per-chunk RMS gain-norm (our recommended swap)

Structure: fixed distortionless reference path w_ref = a/M (spectrally
flat by construction — same DAS weights at every bin) + adaptive noise
canceller acting ONLY on the blocking-matrix error signal.

- **Why spectrally flat:** the target passes through the FIXED reference
  path; the adaptive filter multiplies only the error (noise) path.
  Measured line-gain spread 0.21 dB (vs LCMV 3.2, PhaseOnly 3.9); AST
  keeps the class ("Speech"/"Siren") under phase error.
- **Why phase/gain robust:** the reference path never inverts R, so
  mismatch degrades it as slowly as DAS itself; measured 12.2 dB SI-SDR
  at +-20 deg (LCMV 1.3). Gain errors are handled by the explicit RMS
  normalization (provably exact for far-field-dominated scenes).
- **Known limit:** the blocking matrix leakage at extreme INR — leaks a
  +40..50 dB close interferer (mission case: helicopter; measured bird
  margins negative). Also per-chunk normalization MUST NOT feed the
  level-history path (norm-ruin: -15..-30 dB onset artifacts).

## 4. SubspaceNull (extreme-INR specialist; second stage candidate)

Projects out the k dominant eigenvectors of R(f), then filters the
residual toward the look direction.

- **Why it survives +40..50 dB interference:** the loud interferer IS the
  dominant eigendirection; removing it does not require the steering to
  be accurate, only the covariance — which the loud source itself
  estimates well. Near-field rank-2 spread is why k=2.
- Limits: needs the target BELOW the loud subspace (fine for the mission:
  weak bird vs loud helicopter — the only dual-LAM-positive method,
  AST +0.19 / YAMNet +0.55); moderate-phase robustness, weaker than GSC.

## 5. Decision table (all numbers from committed CSVs)

| Property | LCMV | PhaseOnly | GSC+gn | SubspaceNull |
|---|---|---|---|---|
| Gain-error immunity | none | **provably yes** | via RMS-norm | partial |
| Phase-error immunity | none (0.1 dB @45deg) | none (phase-blind) | **12.2 dB @20deg** | partial |
| Spectral fidelity (line spread) | 3.2 dB | 3.9 dB | **0.21 dB** | ~1-3 dB |
| Null depth / rejection | best calibrated (dSIR 44) | shallow (~-13 dB SSR) | good (34) | best under extreme INR |
| Extreme INR (+40..50) | dies | dies | leaks | **survives (bird margins +)** |
| AST class preservation (field) | Vehicle | Speech | **Speech** | Speech |
| History stability (their anomaly stat) | ok | worst (1.1-1.3 dB) | **0.33 dB** | ok |
| Onset artifact with per-chunk norms | n/a | n/a | avoid via dual-path rule | n/a |

Recommendation stands: GSC+gn (with dual-path rule) as the swap;
SubspaceNull added at beams where a loud interferer is detected; PhaseOnly
as the fallback when no gain normalization is allowed at all — its
invariance is real, but it pays in null depth and keeps the coloration.

Roadmap (user-set): (1) grid-swap now, (2) BSS (IVA/ILRMA) AFTER the grid
beamformer on its outputs, (3) full pipeline comparison: grid-swap vs
DOA-first vs BSS+beamformer hybrid.

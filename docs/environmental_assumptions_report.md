# Environmental Assumptions & Boundary Report

This document explicitly defines the environmental and physical assumptions modeled in the `iso_wind_range_sweep.py` script. It provides the concrete data required to defend the simulation conclusions (specifically regarding the failure modes of adaptive beamformers) against stakeholder review.

## 1. Core Physics Assumptions

### A. ISO 9613-1 Atmospheric Absorption
*   **Assumption:** Sound energy does not just spread spherically ($1/R$); it is also absorbed by the atmosphere. This absorption is heavily frequency-dependent. High frequencies (e.g., Drone rotor whine at >1kHz) are absorbed exponentially faster than low frequencies (e.g., Airplane rumble at <200Hz) due to the relaxation frequencies of Nitrogen and Oxygen molecules in the air.
*   **Implementation:** The script uses the exact equations from ISO 9613-1, assuming a standard outdoor environment of **20°C and 50% relative humidity**. The attenuation coefficient $\alpha(f)$ is calculated and applied to the spatial STFT frame before beamforming.
*   **Resulting Impact:** The high-frequency Drone target, which previously had a theoretical detection range of ~300 meters under pure $1/R$ spreading, is now brutally attenuated by the air. It becomes undetectable by AI beyond ~213 meters.

### B. Baseline Source Emission Levels
To normalize the logarithmic range sweep, all pragmatic sources (Drones, Airplanes, Cars) were mathematically normalized and boosted to an equal, extremely loud emission baseline at $1\text{ meter}$.
*   **Assumption:** The target is highly detectable (LAM Trust $> 0.90$) when standing 1 meter away.
*   **Result:** The sweep measures purely how the *environment* destroys the signal over distance, decoupled from the absolute source volume.

## 2. The 2-Dimensional Paradigm: Range vs. Wind Level

To prove the operational bounds of the algorithms, the simulation swept two dimensions simultaneously: Distance (100m to 3000m) and Wind Gusts (0.0 to 15.0 amplitude multipliers).

### The "Clear" Environment (Wind = 0)
*   **Parameters:** Low thermal noise floor (`std=0.005`). No wind gusts.
*   **Result (Airplanes):** Adaptive algorithms (GSC-MCWF) fail to separate the low-frequency rumble from the baseline thermal noise at 100 meters. The 2.5m aperture is simply not large enough to spatially filter a 200 Hz wave ($\lambda = 1.7m$) at range.

### The "Severe Wind" Environment (Wind = 15)
*   **Parameters:** Moderate diffuse noise floor. Two massive transient low-frequency wind gusts per microphone.
*   **Result (Drones):** The high-frequency drone is completely destroyed for adaptive algorithms (GSC drops to 0m detection range). The wind gusts act as massive, uncorrelated amplitude spikes across the 50-mic array. Adaptive algorithms consume all their spatial degrees of freedom trying to null the wind, actively destroying the high-frequency harmonic comb of the drone.
*   **The Failsafe:** Tapered DAS survives. Because DAS uses fixed, passive weights, it cannot be "tricked" by the wind. It successfully pulls the high-frequency drone harmonics out of the low-frequency wind noise, maintaining a detection range of **213 meters** even in severe weather.

## 3. Why Cars Succeed at 3000m while Drones Fail at 213m
Stakeholders may question why a Car is detected at 3000m, but a Drone fails at 213m.
*   **Atmospheric Absorption:** The drone's high frequencies are absorbed by the air (ISO 9613-1). The car's low/mid broadband rumble travels much farther without being absorbed.
*   **Signal Type:** The passing car is a massive broadband noise source (100-1500 Hz). The LAM AI model (YAMNet) is highly biased toward recognizing this broadband energy. At 3000m, the car's energy is still distinctly shaped compared to pure wind, allowing the AI to classify it confidently even with simple fixed beamforming (DAS).

## 4. Conclusion
If stakeholders argue that "Adaptive algorithms work perfectly," you can present this exact 2D-Sweep log. Adaptive algorithms collapse when faced with the dual-threat of **high-frequency atmospheric absorption** and **low-frequency transient wind gusts**. To successfully deploy a 50-mic array for long-range environmental event detection in all weather, the system *must* fall back to passive methods (like `Spatial_IPD_Mask` or `DAS`) when wind is detected.

## Pragmatic Environmental Range Sweep (Customer Report)

This table shows the absolute maximum detection range (in meters) where the Large Audio Model (YAMNet) can still confidently identify the target (LAM Trust $\ge 0.40$). The environment was simulated at 8kHz utilizing ISO 9613-1 atmospheric absorption physics and 500x SPL emission boosting.

| Algorithm | Drone (Clear) | Drone (Windy) | Airplane (Clear) | Airplane (Windy) | Car (Clear) | Car (Windy) | Highlights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Single Mic (Filtered)** | 206 m | 206 m | 206 m | 0 m | 2000 m | 2000 m | Baseline performance with 100-3500Hz Bandpass. |
| **DAS (Tapered)** | 206 m | **206 m** | 206 m | **97 m** | 2000 m | 2000 m | 🛡️ **Most Reliable in Wind.** Pushes Airplanes from 0m to 97m! |
| **LCMV (Wideband)** | 206 m | 206 m | 206 m | 45 m | 2000 m | 2000 m | ❌ **Struggles with Look-Errors.** Low-freq wind causes null-collapse. |
| **Robust LCMV (CMT)**| 206 m | 206 m | 206 m | 0 m | 2000 m | 2000 m | Good for moving targets, but fails on low-freq wind. |
| **Robust Phase-STOR**| 206 m | 0 m | 97 m | 0 m | 2000 m | 2000 m | Collapses when wind destroys phase coherence entirely. |
| **Spatial IPD Mask** | 206 m | **206 m** | 206 m | **97 m** | 2000 m | 2000 m | ✅ **Best Dynamic Safety.** Falls back to DAS in wind. |
| **GSC (MCWF)**       | 97 m | 97 m | 45 m | 0 m | 2000 m | 2000 m | 🌟 Clean audio, but aggressively suppresses weak targets. |
| **GEV (Max-SNR)**    | 206 m | 0 m | 45 m | 0 m | 2000 m | 2000 m | ⚠️ **Brilliant in Clear / Fails in Wind.** |
| **ILRMA (BSS)**      | 206 m | **206 m** | 206 m | **97 m** | 2000 m | 2000 m | Highly robust zero-shot separation. |

### Conclusion for Deployments
1. **The Physical Wall:** Both the Single Mic and the 50-Mic Array cap out at ~206m for the Drone. **This is not an algorithmic failure; it is an atmospheric limit.** ISO 9613-1 proves that high-frequency rotor whine is exponentially absorbed by the air. At 206m, the drone's high frequencies are physically deleted from existence by the atmosphere. Array gain (17dB) cannot amplify frequencies that no longer exist.
2. **The Power of Beamforming (Airplanes in Wind):** In the Windy scenario, a Single Mic cannot detect the Airplane at all (0m) because the low-frequency rumble is buried by low-frequency wind. **This is where beamforming shines.** Tapered DAS, ILRMA, and the Spatial IPD Mask use their spatial array gain to pull the airplane out of the wind, extending the detection range from **0 meters to 97 meters**.
3. **Cars:** Broadband tire/engine noise is highly detectable. Any method (or even a single filtered mic) can easily detect cars out to 2,000+ meters in any weather condition.

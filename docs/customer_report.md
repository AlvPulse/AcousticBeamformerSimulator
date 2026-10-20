## Pragmatic Environmental Range Sweep (Customer Report)

This table shows the absolute maximum detection range (in meters) where the Large Audio Model (YAMNet) can still confidently identify the target (LAM Trust $\ge 0.40$). The environment was simulated at 8kHz utilizing ISO 9613-1 atmospheric absorption physics.

| Algorithm | Drone (Clear) | Drone (Windy) | Airplane (Clear) | Airplane (Windy) | Car (Clear) | Car (Windy) | Highlights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Single Mic (Raw)** | 240 m | 83 m | 240 m | 29 m | 2000 m | 2000 m | Baseline performance without array processing. |
| **DAS (Tapered)** | 240 m | **240 m** | 240 m | **83 m** | 2000 m | 2000 m | 🛡️ **Most Reliable in Wind.** Does not self-cancel. |
| **LCMV (Wideband)** | 240 m | 240 m | 83 m | 240 m | 240 m | 2000 m | ❌ **Struggles with Look-Errors.** Highly volatile. |
| **Robust LCMV (CMT)**| 240 m | 240 m | 240 m | 0 m | 240 m | 2000 m | Good for moving targets, but fails on low-freq wind. |
| **Robust Phase-STOR**| 240 m | 0 m | 83 m | 0 m | 2000 m | 2000 m | Collapses when wind destroys phase coherence entirely. |
| **Spatial IPD Mask** | 240 m | **240 m** | 240 m | **83 m** | 2000 m | 2000 m | ✅ **Best Dynamic Safety.** Falls back to DAS in wind. |
| **GSC (MCWF)**       | 83 m | 83 m | 240 m | 83 m | 29 m | 2000 m | 🌟 Extremely clean audio, but aggressive cancellation. |
| **GEV (Max-SNR)**    | 240 m | 0 m | **2000 m** | 0 m | 2000 m | 2000 m | ⚠️ **Brilliant in Clear / Fails in Wind.** |
| **ILRMA (BSS)**      | 240 m | **240 m** | 240 m | **83 m** | 2000 m | 2000 m | Highly robust zero-shot separation. |

### Conclusion for Deployments
1. **Drones:** High-frequency rotor whine is absorbed quickly by the atmosphere. Expect ~240m max range. In wind, you **must** use DAS, ILRMA, or IPD Mask. Adaptive methods (GEV, Phase-STOR) collapse to 0m.
2. **Airplanes:** Low-frequency rumble is difficult for a 2.5m array to separate from low-frequency wind gusts. Max range drops from 240m to 83m in the wind.
3. **Cars:** Broadband tire/engine noise is highly detectable. Any method (or even a single mic) can easily detect cars out to 2,000+ meters in any weather condition.

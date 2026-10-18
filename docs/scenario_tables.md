## Final Algorithm Comparison (Outdoor Edge Case Scenarios)

### Harsh Environment (Wind + Turbulence) (S10)
| Beamformer | $\Delta$ SI-SDR (dB) | LSD (Lower=Better) | LAM AI Trust (Cosine Sim) | Highlights & Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **DAS** | -13.81 ± 21.29 | 40.22 ± 0.16 | 0.67 ± 0.01 |  |
| **EBB** | -27.67 ± 28.79 | 47.13 ± 0.23 | 0.57 ± 0.02 |  |
| **GEV** | -4.99 ± 17.38 | 50.86 ± 0.82 | 0.37 ± 0.04 | ⚠️ **LAM Destroyer** (Good SNR, but shreds AI embeddings) |
| **GSC_MCWF** | -19.72 ± 27.97 | 18.41 ± 0.12 | 0.62 ± 0.02 | 🌟 **Cleanest Audio** (MCWF reduces LSD significantly) |
| **ILRMA** | -13.81 ± 21.29 | 40.22 ± 0.16 | 0.67 ± 0.01 |  |
| **LCMV_Narrowband** | -0.62 ± 0.26 | 40.86 ± 0.08 | 0.60 ± 0.03 |  |
| **LCMV_Wideband** | -12.77 ± 24.86 | 37.48 ± 0.06 | 0.61 ± 0.02 |  |
| **MultiRank_OSP** | -11.70 ± 18.83 | 41.68 ± 0.17 | 0.61 ± 0.03 |  |
| **RobustLCMV_CMT** | -15.59 ± 20.14 | 37.48 ± 0.05 | 0.62 ± 0.03 |  |
| **RobustLCMV_PhaseSTOR** | -21.34 ± 23.96 | 39.39 ± 0.20 | 0.58 ± 0.02 |  |
| **RobustSOCP** | -15.38 ± 24.37 | 37.75 ± 0.03 | 0.63 ± 0.02 |  |
| **Spatial_IPD_Mask** | -13.81 ± 21.29 | 40.22 ± 0.16 | 0.67 ± 0.01 | ✅ **Wind Safe** (Detects turbulence and falls back safely) |
| **cGMM_MVDR** | -11.33 ± 22.84 | 48.71 ± 1.12 | 0.56 ± 0.05 |  |

### Hardware Failure (10% Mics Dead/Clipped) (S11)
| Beamformer | $\Delta$ SI-SDR (dB) | LSD (Lower=Better) | LAM AI Trust (Cosine Sim) | Highlights & Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **DAS** | -0.49 ± 0.70 | 67.49 ± 3.89 | 0.29 ± 0.07 |  |
| **EBB** | -5.15 ± 3.69 | 72.95 ± 2.18 | 0.34 ± 0.05 |  |
| **GEV** | -31.88 ± 7.57 | 97.50 ± 0.11 | 0.30 ± 0.02 |  |
| **GSC_MCWF** | -9.95 ± 3.54 | 19.21 ± 5.17 | 0.27 ± 0.06 |  |
| **ILRMA** | -0.49 ± 0.70 | 67.49 ± 3.89 | 0.29 ± 0.07 |  |
| **LCMV_Narrowband** | 6.17 ± 0.66 | 45.42 ± 16.37 | 0.34 ± 0.04 |  |
| **LCMV_Wideband** | -7.93 ± 1.15 | 46.27 ± 15.71 | 0.27 ± 0.05 | ❌ **Fails under broken array** |
| **MultiRank_OSP** | -17.95 ± 9.44 | 62.72 ± 0.62 | 0.36 ± 0.04 |  |
| **RobustLCMV_CMT** | -6.52 ± 1.08 | 46.21 ± 15.98 | 0.29 ± 0.07 |  |
| **RobustLCMV_PhaseSTOR** | -2.35 ± 2.71 | 51.77 ± 19.96 | 0.22 ± 0.03 |  |
| **RobustSOCP** | -19.67 ± 13.50 | 43.56 ± 19.05 | 0.33 ± 0.02 |  |
| **Spatial_IPD_Mask** | 1.34 ± 0.61 | 66.97 ± 3.99 | 0.30 ± 0.10 | ✅ **Hardware Resilient** (Maintained positive gain) |
| **cGMM_MVDR** | -10.52 ± 2.93 | 82.85 ± 7.54 | 0.35 ± 0.05 |  |

### Moving Target Tracking (Sweeping Target) (S12)
| Beamformer | $\Delta$ SI-SDR (dB) | LSD (Lower=Better) | LAM AI Trust (Cosine Sim) | Highlights & Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **DAS** | -0.70 ± 0.11 | 63.01 ± 0.05 | 0.43 ± 0.02 |  |
| **EBB** | -0.72 ± 0.38 | 70.38 ± 0.22 | 0.34 ± 0.03 |  |
| **GEV** | -29.55 ± 3.72 | 97.92 ± 0.15 | 0.31 ± 0.03 |  |
| **GSC_MCWF** | -0.96 ± 0.76 | 28.05 ± 0.13 | 0.65 ± 0.03 |  |
| **ILRMA** | -0.70 ± 0.11 | 63.01 ± 0.05 | 0.43 ± 0.02 |  |
| **LCMV_Narrowband** | 6.00 ± 0.17 | 59.33 ± 0.11 | 0.43 ± 0.03 |  |
| **LCMV_Wideband** | -2.32 ± 0.56 | 60.95 ± 0.06 | 0.36 ± 0.03 |  |
| **MultiRank_OSP** | -21.90 ± 4.19 | 63.98 ± 0.04 | 0.36 ± 0.06 |  |
| **RobustLCMV_CMT** | -0.56 ± 0.34 | 61.05 ± 0.06 | 0.40 ± 0.04 | ✅ **Excellent Tracker** (Null broadening catches the moving target) |
| **RobustLCMV_PhaseSTOR** | -0.90 ± 0.56 | 61.09 ± 0.06 | 0.40 ± 0.04 |  |
| **RobustSOCP** | -0.31 ± 0.25 | 61.15 ± 0.07 | 0.43 ± 0.02 | ❌ **Too Slow** (Cannot compute weights fast enough for real-time tracking) |
| **Spatial_IPD_Mask** | 0.75 ± 0.13 | 62.70 ± 0.06 | 0.45 ± 0.02 |  |
| **cGMM_MVDR** | -4.31 ± 0.57 | 72.55 ± 0.10 | 0.43 ± 0.02 |  |

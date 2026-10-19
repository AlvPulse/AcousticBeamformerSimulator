import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
import os
import sys

# Ensure simulator imports work
sys.path.append('.')
from simulator.beamformers import TaperedDAS, LCMV_SMI, RobustLCMV, GSC_MCWF, ILRMA, SpatialVarianceMask

def run_extreme_demo():
    print("==================================================")
    print("Executing 100-Mic 2Km Extreme Range Demo")
    print("==================================================")

    # 1. SCENARIO SETUP
    fs = 16000
    T = 8.0
    N = int(fs * T)
    t = np.arange(N) / fs
    c = 343.0

    # 10x10 Microphone Array (4.5m x 4.5m)
    x = np.linspace(-2.25, 2.25, 10)
    y = np.linspace(-2.25, 2.25, 10)
    X, Y = np.meshgrid(x, y)
    mics = np.column_stack((X.ravel(), Y.ravel(), np.zeros(100)))
    M = mics.shape[0]

    # Target: Bird at 2000m (Azimuth 90 degrees if looking down Y axis,
    # but based on Jules script: [0, 2000, 10] -> Azimuth 90 deg, Elevation ~0 deg)
    target_az = 90.0
    target_el = 0.0
    target_pos = np.array([0.0, 2000.0, 10.0])
    dist_t = np.linalg.norm(mics - target_pos, axis=1)

    # Interferer: Car moving at 100m (x: -50 to 50, y: 100, speed: 20 m/s)
    car_x = -50.0 + 20.0 * t
    car_y = 100.0
    car_z = 0.0

    # 2. SIGNAL GENERATION
    # Bird Call: 2-4 kHz FM chirp with Hann envelope, active 5.5s to 7.5s
    bird_env = np.zeros(N)
    start_idx = int(5.5 * fs)
    end_idx = int(7.5 * fs)
    length = end_idx - start_idx
    bird_env[start_idx:end_idx] = np.hanning(length)

    bird_src = np.sin(2 * np.pi * (2000 + 500 * (t - 5.5)) * (t - 5.5))
    bird_src = bird_src * bird_env

    # Car Noise: Broadband 100-1500 Hz, active 1.0s to 6.0s
    car_env = np.where((t >= 1.0) & (t <= 6.0), 1.0, 0.0)
    car_white = np.random.randn(N)
    b, a = signal.butter(4, [100/(fs/2), 1500/(fs/2)], btype='band')
    car_src = signal.filtfilt(b, a, car_white) * car_env * 50.0

    # Wind Noise: Lowpass < 150 Hz + Gusts at 2.0s and 6.5s
    wind_white = np.random.randn(M, N)
    b_w, a_w = signal.butter(2, 150/(fs/2), btype='low')
    wind_noise = np.array([signal.filtfilt(b_w, a_w, w) for w in wind_white])
    gusts = 8.0 * np.exp(-((t - 2.0)**2) / 0.15) + 10.0 * np.exp(-((t - 6.5)**2) / 0.15)
    wind_noise = wind_noise * (1.0 + gusts)

    # 3. PROPAGATION TO ARRAY
    print("Simulating 100-microphone propagation...")
    Y_mics = np.zeros((M, N))

    for m in range(M):
        # Target (Fractional Delay via interpolation)
        t_delayed = t - (dist_t[m] / c)
        valid = (t_delayed >= 0) & (t_delayed < T)
        Y_mics[m, valid] += np.interp(t_delayed[valid], t, bird_src) / dist_t[m]

        # Car (Time-Varying Fractional Delay)
        car_pos_t = np.column_stack((car_x, np.full(N, car_y), np.full(N, car_z)))
        dist_c = np.linalg.norm(mics[m] - car_pos_t, axis=1)
        t_delayed_c = t - (dist_c / c)
        valid_c = (t_delayed_c >= 0) & (t_delayed_c < T)
        Y_mics[m, valid_c] += np.interp(t_delayed_c[valid_c], t, car_src) / dist_c[valid_c]

        # Wind
        Y_mics[m, :] += wind_noise[m, :]

    print("Propagation complete.")

    # 4. BEAMFORMING EVALUATION
    # We explicitly use the verified, STFT-based math from the simulator
    beamformers = {
        "Single_Mic": None, # Reference
        "Tapered_DAS": TaperedDAS(fs=fs),
        "LCMV_Wideband": LCMV_SMI(fs=fs),
        "RobustLCMV_CMT": RobustLCMV(fs=fs, use_cmt=True),
        "RobustLCMV_PhaseSTOR": RobustLCMV(fs=fs, use_phase_stor=True),
        "GSC_MCWF": GSC_MCWF(fs=fs),
        "Spatial_IPD_Mask": SpatialVarianceMask(fs=fs),
        "ILRMA": ILRMA(fs=fs, n_iter=10)
    }

    outputs = {}

    out_dir = "docs/audio_outputs/demo_2km"
    os.makedirs(out_dir, exist_ok=True)

    print("\nProcessing Beamformers (STFT Math)...")
    for name, bf in beamformers.items():
        print(f" -> {name}...")
        if name == "Single_Mic":
            y_out = Y_mics[50, :] # Center mic
        else:
            y_out = bf.process(Y_mics, mics, target_az, target_el)

        # Normalize to prevent clipping in WAV
        y_out = y_out / (np.max(np.abs(y_out)) + 1e-10)
        outputs[name] = y_out

        # Save to disk
        sf.write(f"{out_dir}/{name}.wav", y_out, fs)

    # 5. VISUALIZATION
    print("\nGenerating Comparison Spectrograms...")
    fig, axes = plt.subplots(len(beamformers), 1, figsize=(12, 3 * len(beamformers)))

    # Target is active 5.5 to 7.5s, Car is active 1.0 to 6.0s
    # We plot the full duration to see the car approach and wind gusts.

    for i, (name, y_out) in enumerate(outputs.items()):
        ax = axes[i]
        Pxx, freqs, bins, im = ax.specgram(y_out, NFFT=512, Fs=fs, noverlap=256, cmap='magma', vmin=-100, vmax=0)
        ax.set_title(f'{name} (Target: 2-4kHz @ 5.5s-7.5s | Wind Gusts @ 2s, 6.5s)')
        ax.set_ylabel('Freq (Hz)')
        if i == len(beamformers) - 1:
            ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig("docs/figures/beamforming_comparison_2km.png", dpi=300)
    print("Saved: docs/figures/beamforming_comparison_2km.png")
    print("Done!")

if __name__ == "__main__":
    run_extreme_demo()

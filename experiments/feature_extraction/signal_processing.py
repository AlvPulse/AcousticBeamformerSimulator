import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from multiprocessing import Pool
from experiments.feature_extraction import EncoderConfig
import os

def load_audio(filepath, target_fs=EncoderConfig.SAMPLE_RATE):
    try:
        fs, audio = wavfile.read(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    if fs != target_fs:
        num_samples = int(len(audio) * target_fs / fs)
        audio = signal.resample(audio, num_samples)
        fs = target_fs

    return audio, fs

def compute_psd(audio, fs, nperseg=EncoderConfig.N_FFT):
    f, Pxx = signal.welch(audio, fs, nperseg=nperseg)
    Pxx_db = 10 * np.log10(Pxx + 1e-10)
    return f, Pxx_db

# Helper function for multiprocessing
def _process_frame(args):
    psd_frame, f_valid, start_f_idx, min_prominence, min_dist, max_peaks = args
    p_idxs, properties = signal.find_peaks(psd_frame, prominence=min_prominence, distance=min_dist)

    detected_peaks = []
    if len(p_idxs) > 0:
        prominences = properties['prominences']
        powers = psd_frame[p_idxs]
        freqs_found = f_valid[p_idxs]

        sort_order = np.argsort(powers)[::-1][:max_peaks]

        for idx in sort_order:
            detected_peaks.append({
                'freq': freqs_found[idx],
                'power': powers[idx],
                'snr': prominences[idx],
                'idx': p_idxs[idx] + start_f_idx
            })

        detected_peaks.sort(key=lambda x: x['freq'])

    return detected_peaks

def compute_spectrogram_and_peaks(audio, fs, nperseg=EncoderConfig.N_FFT, noverlap=None, parallel=False):
    if noverlap is None:
        if nperseg < EncoderConfig.N_FFT:
            noverlap = nperseg // 2
        else:
            noverlap = nperseg - EncoderConfig.HOP_LENGTH

    f, t, Zxx = signal.stft(audio, fs, nperseg=nperseg, noverlap=noverlap)
    Pxx = np.abs(Zxx)**2
    Pxx_db = 10 * np.log10(Pxx + 1e-10)

    valid_mask = (f >= EncoderConfig.MIN_FREQ) & (f <= EncoderConfig.MAX_FREQ)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return f, t, Pxx_db, [[] for _ in t]

    start_f_idx = valid_indices[0]
    end_f_idx = valid_indices[-1] + 1

    f_valid = f[start_f_idx:end_f_idx]
    Pxx_db_valid = Pxx_db[start_f_idx:end_f_idx, :]

    min_prominence = EncoderConfig.PEAK_PROMINENCE
    min_dist = EncoderConfig.PEAK_DISTANCE
    max_peaks = EncoderConfig.NUM_PEAKS_TO_KEEP

    num_frames = Pxx_db_valid.shape[1]

    if parallel and num_frames > 10:
        pool_args = [
            (Pxx_db_valid[:, i], f_valid, start_f_idx, min_prominence, min_dist, max_peaks)
            for i in range(num_frames)
        ]
        num_cores = min(os.cpu_count() or 1, 8)
        with Pool(processes=num_cores) as pool:
            peaks_per_frame = pool.map(_process_frame, pool_args)
    else:
        peaks_per_frame = []
        for i in range(num_frames):
            peaks_per_frame.append(
                _process_frame((Pxx_db_valid[:, i], f_valid, start_f_idx, min_prominence, min_dist, max_peaks))
            )

    return f, t, Pxx_db, peaks_per_frame

def clean_multichannel_audio(audio, fs=16000, cutoff_hz=200.0, taper_ms=50):
    """
    Cleans raw multichannel array data:
    1. Removes static DC mean per channel.
    2. Zero-phase high-pass filter (no phase shift).
    3. Tapers boundary edges to suppress startup/teardown transients.
    """
    # 1. Demean each channel
    audio = audio - np.mean(audio, axis=1, keepdims=True)

    # 2. Zero-phase High-Pass Filter (4th-order Butterworth)
    sos = signal.butter(4, cutoff_hz, btype='highpass', fs=fs, output='sos')
    audio = signal.sosfiltfilt(sos, audio, axis=1)

    # 3. Smooth fade-in and fade-out (Tukey window taper)
    taper_samples = int((taper_ms / 1000.0) * fs)
    if taper_samples > 0 and audio.shape[1] > 2 * taper_samples:
        fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
        audio[:, :taper_samples] *= fade
        audio[:, -taper_samples:] *= fade[::-1]

    return audio

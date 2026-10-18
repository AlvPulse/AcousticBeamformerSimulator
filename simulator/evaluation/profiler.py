import time
import os

class Profiler:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, audio_duration_sec):
        self.end_time = time.time()
        process_time = self.end_time - self.start_time
        rtf = process_time / audio_duration_sec

        # Calculate absolute latency per STFT block (Assuming 512 n_fft, 256 hop @ 16kHz -> 16ms blocks)
        # Total blocks = (audio_duration_sec * fs) / hop_length
        fs = 16000
        hop_length = 256
        total_blocks = (audio_duration_sec * fs) / hop_length
        latency_ms_per_block = (process_time / total_blocks) * 1000.0

        return {"process_time": process_time, "rtf": rtf, "latency_ms": latency_ms_per_block}

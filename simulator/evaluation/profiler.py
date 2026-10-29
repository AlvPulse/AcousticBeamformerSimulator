import time
import tracemalloc

class Profiler:
    def __init__(self, fs):
        self.fs = fs
        self.start_time = 0
        self.end_time = 0
        self.audio_duration = 0

    def start(self, n_samples):
        self.audio_duration = n_samples / self.fs
        tracemalloc.start()
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        processing_time = self.end_time - self.start_time
        rtf = processing_time / self.audio_duration

        return {
            "processing_time_s": processing_time,
            "rtf": rtf,
            "peak_memory_mb": peak / 10**6
        }

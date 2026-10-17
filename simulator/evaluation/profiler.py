import time
import psutil
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
        return {"process_time": process_time, "rtf": rtf}

import numpy as np

# Global simulation parameters
FS = 16000
N_FFT = 512
HOP_LENGTH = 256
C = 343.0  # Speed of sound

ARRAY_DIAMETER = 2.5
N_MICS = 50

# Source configs
FAR_FIELD_DIST = 120.0
NEAR_FIELD_DIST_RANGE = (6.0, 20.0)
CAR_VELOCITY_MPS = 15.0  # 54 km/h

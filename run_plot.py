import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath('.'))
from experiments.generate_signature_plots import generate_harmonic_fpv_scenario

out_dir = os.path.abspath('.')
generate_harmonic_fpv_scenario(out_dir)

from acoustic_sim.dsp_array import load_array
import numpy as np

# Create a mock CSV matching the screenshot format
with open("test_array.csv", "w") as f:
    f.write("0.231164,0.363408,0.337021\n")
    f.write("0.160843,0.146471,0.284884\n")
    f.write("0,0,0\n")

arr = load_array("test_array.csv")
print("Parsed X:", arr.x + np.mean([0.231164, 0.160843, 0]))  # Adding back the mean because ArrayGeometry centers them
print("Parsed Y:", arr.y + np.mean([0.363408, 0.146471, 0]))

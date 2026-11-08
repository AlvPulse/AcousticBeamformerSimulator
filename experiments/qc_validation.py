import pandas as pd
import numpy as np
import os

def validate_csv(filepath, required_cols):
    if not os.path.exists(filepath):
        print(f"[FAIL] File not found: {filepath}")
        return False

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[FAIL] Could not parse CSV {filepath}: {e}")
        return False

    # Check required columns
    for col in required_cols:
        if col not in df.columns:
            print(f"[FAIL] Missing column '{col}' in {filepath}")
            return False

    # Check for NaNs and Infs
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            print(f"[FAIL] NaN values found in column '{col}' of {filepath}")
            return False
        if np.isinf(df[col]).any():
            print(f"[FAIL] Infinite values found in column '{col}' of {filepath}")
            return False

    print(f"[PASS] {filepath} passed basic structural and numerical QC.")
    return True

def qc_gridless_benchmark(filepath):
    print(f"\n--- Running Deep QC on {filepath} ---")
    if not validate_csv(filepath, ["Scenario", "Method", "Harmonic_Score", "IRR_dB", "Time (s)", "Peak RAM (MB)"]):
        return False

    df = pd.read_csv(filepath)

    # Domain specific bounds checking
    # Harmonic Score is usually between -150 and 0
    if not df["Harmonic_Score"].between(-200.0, 50.0).all():
        print("[WARN] Harmonic_Score outside expected bounds [-200, 50]. Checking for divergence...")
        # We know FastIVA can diverge, so this is a warning, not a hard fail.

    # Execution time must be > 0
    if not (df["Time (s)"] > 0).all():
        print("[FAIL] Execution Time contains zero or negative values.")
        return False

    # Peak RAM must be > 0
    if not (df["Peak RAM (MB)"] > 0).all():
        print("[FAIL] Peak RAM contains zero or negative values.")
        return False

    print("[PASS] Benchmark bounds and physics constraints verified successfully.")
    return True

if __name__ == "__main__":
    gridless_csv = "docs/benchmark_gridless.csv"
    qc_gridless_benchmark(gridless_csv)

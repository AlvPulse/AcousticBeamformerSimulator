import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_diagnostics(csv_path="docs/benchmark_results.csv", output_dir="docs/figures"):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Run benchmark first.")
        return

    df = pd.read_csv(csv_path)

    # 1. Array Gain vs Beamformer
    plt.figure(figsize=(12, 6))
    ag_data = df.groupby('Beamformer')['Array_Gain_dB'].mean().sort_values()

    ag_data.plot(kind='bar', color='steelblue')
    plt.title("Coherent Combining Analysis: True Array Gain (SNR_out - SNR_in)")
    plt.ylabel("Mean Array Gain (dB)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/array_gain_analysis.png")
    plt.close()

    # 2. LAM Preservation vs Beamformer
    plt.figure(figsize=(12, 6))
    lam_data = df.groupby('Beamformer')['LAM_CosineSim'].mean().sort_values()

    colors = ['firebrick' if val < 0.4 else 'darkorange' if val < 0.6 else 'forestgreen' for val in lam_data]

    ax = lam_data.plot(kind='bar', color=colors)
    plt.axhline(y=0.6, color='g', linestyle='--', label='Trust Threshold')
    plt.axhline(y=0.4, color='r', linestyle='--', label='Fail Threshold')
    plt.fill_between([-1, len(lam_data)], 0.6, 1.0, color='g', alpha=0.1)
    plt.fill_between([-1, len(lam_data)], 0.0, 0.4, color='r', alpha=0.1)

    plt.title("Decision Matrix: Downstream LAM Feature Integrity")
    plt.ylabel("YAMNet Cosine Embedding Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lam_decision_matrix.png")
    plt.close()

if __name__ == "__main__":
    plot_diagnostics()

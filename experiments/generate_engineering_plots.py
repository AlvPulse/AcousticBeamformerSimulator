import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    csv_path = "docs/benchmark_gridless.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs("docs/figures", exist_ok=True)

    # 1. Compute vs Quality Trade-off (Scatter Plot)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="Time (s)",
        y="Harmonic_Score",
        hue="Method",
        style="Scenario",
        s=200,
        palette="tab10"
    )
    plt.title("Performance vs. Computation Time (Pareto Frontier)", fontsize=14, fontweight='bold')
    plt.xlabel("Execution Time (Seconds) - Lower is Better", fontsize=12)
    plt.ylabel("Harmonic Score (dB) - Higher is Better", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("docs/figures/compute_vs_quality.png")
    plt.close()

    # 2. Interference Rejection (Bar Chart)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Method",
        y="IRR_dB",
        hue="Scenario",
        palette="viridis"
    )
    plt.title("Interference Rejection Ratio (IRR) across Scenarios", fontsize=14, fontweight='bold')
    plt.xlabel("Beamforming / Separation Method", fontsize=12)
    plt.ylabel("IRR (dB) - Higher means Deeper Nulls", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("docs/figures/interference_rejection.png")
    plt.close()

    # 3. Peak RAM Usage (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Method",
        y="Peak RAM (MB)",
        palette="magma",
        errorbar=None
    )
    plt.title("Peak Memory (RAM) Consumption per Method", fontsize=14, fontweight='bold')
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Peak RAM (MB) - Lower is Better", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("docs/figures/peak_ram_usage.png")
    plt.close()

    print("Successfully generated engineering plots in docs/figures/")

if __name__ == "__main__":
    generate_plots()

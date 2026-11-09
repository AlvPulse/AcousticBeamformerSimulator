import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    csv_path_realworld = "docs/real_world_outputs/real_world_evaluation.csv"
    csv_path_gridless = "docs/benchmark_gridless.csv"

    df = None
    is_real_world = False

    if os.path.exists(csv_path_realworld):
        print(f"Generating plots from real-world evaluation: {csv_path_realworld}")
        import pandas as pd
        df = pd.read_csv(csv_path_realworld)
        # Map headers
        if "Algorithm" in df.columns:
            df["Method"] = df["Algorithm"]
        if "RTF" in df.columns:
            df["Time (s)"] = df["RTF"]
        if "SSR_Depth_dB" in df.columns:
            df["IRR_dB"] = df["SSR_Depth_dB"]
        df["Scenario"] = "Real-World Test"
        df["Peak RAM (MB)"] = 0 # Not tracked in real-world evaluate script
        is_real_world = True

        # Metric mapping
        if "Harmonic_Score" not in df.columns and "LAM_Detectability" in df.columns:
             df["Harmonic_Score"] = df["LAM_Detectability"]

    elif os.path.exists(csv_path_gridless):
        import pandas as pd
        print(f"Generating plots from synthetic gridless benchmark: {csv_path_gridless}")
        df = pd.read_csv(csv_path_gridless)
    else:
        print("Error: No valid evaluation CSV found to plot.")
        return
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

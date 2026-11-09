import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    csv_path = "docs/real_world_outputs/real_world_evaluation.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs("docs/figures", exist_ok=True)

    # 1. Compute vs Quality Trade-off (Scatter Plot)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="RTF",
        y="SSR_Depth_dB",
        hue="Algorithm",
        s=150,
        palette="tab20",
        alpha=0.8
    )
    plt.title("Compute vs Spatial Suppression Ratio (SSR)", fontsize=14, fontweight='bold')
    plt.xlabel("Execution Time / RTF - Lower is Faster", fontsize=12)
    plt.ylabel("SSR Depth (dB)", fontsize=12)
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("docs/figures/compute_vs_quality.png")
    plt.close()

    # 2. Interference Rejection (Bar Chart)
    plt.figure(figsize=(12, 6))
    if 'SSR_Depth_dB' in df.columns:
        sns.barplot(
            data=df,
            x="Algorithm",
            y="SSR_Depth_dB",
            palette="viridis"
        )
        plt.title("Spatial Suppression Ratio (SSR) Depth", fontsize=14, fontweight='bold')
        plt.xlabel("Beamforming Method", fontsize=12)
        plt.ylabel("SSR (dB) - Higher means Deeper Nulls", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("docs/figures/interference_rejection.png")
        plt.close()

    print("Successfully generated engineering plots in docs/figures/")

if __name__ == "__main__":
    generate_plots()

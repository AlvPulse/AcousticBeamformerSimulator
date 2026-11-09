import subprocess
import sys
import pandas as pd
import os

def run_script(script_path):
    print(f"==================================================")
    print(f"🚀 Running {script_path}...")
    print(f"==================================================")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Error: {script_path} failed to execute.")
        sys.exit(result.returncode)

def generate_analyst_report():
    print(f"\n==================================================")
    print(f"📊 Data Analyst Executive Summary")
    print(f"==================================================")
    results_path = "docs/real_world_outputs/real_world_evaluation.csv"

    if not os.path.exists(results_path):
        print(f"⚠️ Warning: Results file {results_path} not found. Cannot generate summary.")
        return

    df = pd.read_csv(results_path)

    if df.empty:
        print("Dataframe is empty.")
        return

    # Basic stats
    print(f"Total Algorithms Evaluated: {len(df)}")

    # Best Fidelity (SSR_Depth_dB or LAM_Detectability)
    if 'SSR_Depth_dB' in df.columns:
        best_ssr = df.loc[df['SSR_Depth_dB'].idxmax()]
        print(f"🏆 Best Spatial Suppression Ratio (SSR_Depth_dB): {best_ssr['Algorithm']} ({best_ssr['SSR_Depth_dB']:.2f} dB)")

    if 'LAM_Detectability' in df.columns:
        best_lam = df.loc[df['LAM_Detectability'].idxmax()]
        print(f"🏆 Best LAM Detectability / Fidelity: {best_lam['Algorithm']} ({best_lam['LAM_Detectability']:.2f})")

    # Best Real-Time Factor
    if 'RTF' in df.columns:
        best_rtf = df.loc[df['RTF'].idxmin()]
        print(f"⚡ Fastest Execution (Lowest RTF): {best_rtf['Algorithm']} (RTF: {best_rtf['RTF']:.4f})")

    # INCM-MVDR specifically
    incm = df[df['Algorithm'] == 'INCM-MVDR']
    if not incm.empty:
        incm = incm.iloc[0]
        ssr = incm.get('SSR_Depth_dB', 'N/A')
        rtf = incm.get('RTF', 'N/A')
        print(f"🔍 INCM-MVDR Performance -> SSR Depth: {ssr} dB | RTF: {rtf}")

    print("\nSummary table (Top 5 by SSR Depth):")
    if 'SSR_Depth_dB' in df.columns:
        top_5 = df.sort_values(by='SSR_Depth_dB', ascending=False).head(5)
        print(top_5[['Algorithm', 'SSR_Depth_dB', 'RTF', 'LAM_Detectability']].to_markdown(index=False))

    print("\n✅ Pipeline execution and reporting complete.")
    print("Graphs are available in the docs/evaluation_results/ directory.")

if __name__ == "__main__":
    # Ensure dependencies and PYTHONPATH are set
    os.environ["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))

    run_script("experiments/evaluate_real_world.py")
    run_script("experiments/generate_engineering_plots.py")

    generate_analyst_report()

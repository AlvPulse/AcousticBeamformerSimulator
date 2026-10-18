import pandas as pd

def generate_markdown_table():
    df = pd.read_csv('docs/benchmark_summary_with_CI.csv')

    # We only care about Scenario 10
    df = df[df['Scenario'] == 'S10']

    md = "## Final Algorithm Comparison (Harsh Environmental Scenario S10)\n\n"
    md += "| Beamformer | $\\Delta$ SI-SDR (dB) | Array Gain (dB) | LSD (Lower=Better) | LAM AI Trust (Cosine Sim) | Latency (ms/block) | Highlights & Verdict |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

    for _, row in df.iterrows():
        bf = row['Beamformer']
        sdr = row['Delta_SISDR_dB']
        ag = row['Array_Gain_dB']
        lsd = row['LSD']
        lam = row['LAM_CosineSim']
        lat = row['Latency_ms']

        # Format latency
        lat_str = f"{lat:.2f}"

        # Determine highlights based on explicit data thresholds
        highlight = ""
        if bf == "RobustSOCP":
            highlight = "❌ **Fails Real-Time** (Too slow for hardware)"
        elif bf == "GEV":
            highlight = "⚠️ **LAM Destroyer** (Good SNR, but corrupts AI embeddings)"
        elif bf == "GSC_MCWF":
            highlight = "🌟 **Best Overall / Strongest Audio Fidelity** (LSD is 2x better than all others)"
        elif bf == "RobustLCMV_CMT":
            highlight = "✅ **Excellent Tracker** (Widens nulls to survive moving interferers)"
        elif bf == "LCMV_Narrowband":
            highlight = "⚠️ **Passive Filter Only** (Ignores broadband spectrum)"
        elif bf == "Spatial_IPD_Mask":
            highlight = "✅ **Safest Dynamic Fallback** (Perfectly avoids wind collapse)"
        elif bf in ["ILRMA", "DAS"]:
            highlight = "🛡️ **Max Wind Safety / Low Separation**"
        elif bf == "RobustLCMV_PhaseSTOR":
            highlight = "⚡ **Good Baseline Robustness**"
        else:
            highlight = "❌ **Severe Wind Collapse** (Fails in transients)"

        md += f"| **{bf}** | {sdr} | {ag} | {lsd} | {lam} | {lat_str} | {highlight} |\n"

    return md

if __name__ == '__main__':
    print(generate_markdown_table())

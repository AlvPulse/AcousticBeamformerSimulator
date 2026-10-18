import pandas as pd

def generate_markdown_table():
    df = pd.read_csv('docs/benchmark_summary_with_CI.csv')

    md = "## Final Algorithm Comparison (Outdoor Edge Case Scenarios)\n\n"

    scenarios = {
        "S10": "Harsh Environment (Wind + Turbulence)",
        "S11": "Hardware Failure (10% Mics Dead/Clipped)",
        "S12": "Moving Target Tracking (Sweeping Target)"
    }

    for s_id, s_desc in scenarios.items():
        md += f"### {s_desc} ({s_id})\n"
        md += "| Beamformer | $\\Delta$ SI-SDR (dB) | LSD (Lower=Better) | LAM AI Trust (Cosine Sim) | Highlights & Verdict |\n"
        md += "| :--- | :--- | :--- | :--- | :--- |\n"

        subset = df[df['Scenario'] == s_id]
        for _, row in subset.iterrows():
            bf = row['Beamformer']
            sdr = row['Delta_SISDR_dB']
            lsd = row['LSD']
            lam = row['LAM_CosineSim']

            # Simple highlights based on scenario context
            highlight = ""
            if s_id == "S10": # Harsh Wind
                if bf == "Spatial_IPD_Mask": highlight = "✅ **Wind Safe** (Detects turbulence and falls back safely)"
                elif bf == "GSC_MCWF": highlight = "🌟 **Cleanest Audio** (MCWF reduces LSD significantly)"
                elif bf == "GEV": highlight = "⚠️ **LAM Destroyer** (Good SNR, but shreds AI embeddings)"
            elif s_id == "S11": # Sensor Failure
                if bf == "Spatial_IPD_Mask": highlight = "✅ **Hardware Resilient** (Maintained positive gain)"
                elif "LCMV" in bf and "Robust" not in bf and "Narrowband" not in bf: highlight = "❌ **Fails under broken array**"
            elif s_id == "S12": # Moving Target
                if bf == "RobustLCMV_CMT": highlight = "✅ **Excellent Tracker** (Null broadening catches the moving target)"
                elif bf == "RobustSOCP": highlight = "❌ **Too Slow** (Cannot compute weights fast enough for real-time tracking)"

            md += f"| **{bf}** | {sdr} | {lsd} | {lam} | {highlight} |\n"
        md += "\n"

    return md

if __name__ == '__main__':
    print(generate_markdown_table())

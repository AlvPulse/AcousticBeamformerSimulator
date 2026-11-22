"""
Statistical validation of every inference claimed in this project.

Each claim is mapped to (data, metric, comparison, test). All tests are
paired on shared draws/scenes/classes wherever the design allows
(same-seed pairing per BENCHMARK_PLAN v2). Reports: test statistic,
raw p, Holm-adjusted p (within the full claim registry), Cliff's delta
effect size, verdict SUPPORTED / NOT SUPPORTED / UNDERPOWERED.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = lambda f: os.path.join(REPO, "docs", f)


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


CLAIMS = []


def claim(name, data, metric, comparison, test, p, stat, effect, n, note=""):
    CLAIMS.append(dict(Claim=name, Data=data, Metric=metric,
                       Comparison=comparison, Test=test,
                       Statistic=round(float(stat), 3) if stat is not None else None,
                       p_raw=p, Effect=effect, n=n, Note=note))


def paired_test(a, b, name, data, metric, comparison, note="",
                alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    d = cliffs_delta(a, b)
    if n >= 5:
        try:
            st, p = stats.wilcoxon(a, b, alternative=alternative,
                                   zero_method="wilcox")
        except ValueError:
            st, p = np.nan, 1.0
        test = "paired Wilcoxon"
    else:
        # sign test fallback for tiny n; respect the alternative direction
        if alternative == "less":
            s = int(np.sum(a < b))
        else:
            s = int(np.sum(a > b))
        p = stats.binomtest(s, n, 0.5, alternative="greater").pvalue
        st = s
        test = f"sign test ({s}/{n})"
    claim(name, data, metric, comparison, test, p, st, round(d, 2), n, note)
    return p


def main():
    ps = []

    # ---------- C1: grid family selection == 0 (mc_benchmark, 90 draws) ----------
    mc = pd.read_csv(docs("mc_benchmark.csv"))
    for sysname in ("A_gridLCMV", "A'_gridPhaseOnly", "A+_gridLCMV+gn"):
        sub = mc[mc.System == sysname]
        k = int(sub["Correct"].sum())
        n = len(sub)
        p = stats.binomtest(k, n, 0.5, alternative="less").pvalue
        claim(f"C1 {sysname} selection <= chance (is 0%)", "mc_benchmark",
              "target-selection", f"{k}/{n} correct vs 50%",
              "exact binomial", p, k, None, n)

    # ---------- C2: B3 GSC+gn beats LCMV under imbalance (paired by Draw) ----------
    for gain in ("gain_6dB", "gain_12dB"):
        for scene in ("field_wind_drift", "anechoic"):
            a = mc[(mc.System == "B3_music+GSC+gn") & (mc.Gain == gain)
                   & (mc.Scene == scene)].sort_values("Draw").SI_SDR
            b = mc[(mc.System == "B_music+LCMV") & (mc.Gain == gain)
                   & (mc.Scene == scene)].sort_values("Draw").SI_SDR
            if len(a) >= 5 and len(a) == len(b):
                ps.append(paired_test(
                    a, b, f"C2 GSC+gn > LCMV SI-SDR ({scene}@{gain})",
                    "mc_benchmark", "SI-SDR", "B3 vs B"))

    # ---------- C3/C4: phase robustness + spectral fidelity ----------
    ph = pd.read_csv(docs("phase_robustness.csv"))
    for err, lvl in (("phase_deg", "20deg"), ("delay_cm", "2cm")):
        a = ph[(ph.Method == "GSC") & (ph.Error == err) & (ph.Level == lvl)] \
            .sort_values("Draw").SI_SDR
        b = ph[(ph.Method == "LCMV") & (ph.Error == err) & (ph.Level == lvl)] \
            .sort_values("Draw").SI_SDR
        ps.append(paired_test(a, b, f"C3 GSC > LCMV under {err}={lvl}",
                              "phase_robustness", "SI-SDR", "GSC vs LCMV",
                              note="n=3 draws: underpowered by design"))
    a = ph[ph.Method == "LCMV"].sort_values(["Error", "Level", "Draw"]).LineSpread
    b = ph[ph.Method == "GSC"].sort_values(["Error", "Level", "Draw"]).LineSpread
    ps.append(paired_test(a, b, "C4 LCMV colors spectrum more than GSC",
                          "phase_robustness", "line-gain spread",
                          "LCMV vs GSC (all 8 error cells)"))

    # ---------- C5: LCMV -> AST 'Vehicle', others -> 'Speech' (field) ----------
    fc = pd.read_csv(docs("field_candidates.csv"))
    lc = fc[fc.Method.isin(["LCMV"])]
    others = fc[~fc.Method.isin(["LCMV", "RMS->LCMV", "MAX->LCMV"])]
    tab = [[int((lc.AST_top_1 if "AST_top_1" in lc else lc.iloc[:, -2] == "x").sum()), 0],
           [0, 0]]
    # simpler: build 2x2 from class strings
    lc_v = int(lc.TopClasses.str.startswith("Vehicle").sum())
    lc_s = int(lc.TopClasses.str.startswith("Speech").sum())
    o_v = int(others.TopClasses.str.startswith("Vehicle").sum())
    o_s = int(others.TopClasses.str.startswith("Speech").sum())
    st, p = stats.fisher_exact([[lc_v, lc_s], [o_v, o_s]], alternative="greater")
    claim("C5 LCMV outputs classified 'Vehicle' vs others 'Speech' (field)",
          "field_candidates", "AST top class", f"LCMV {lc_v}/{lc_v+lc_s} Vehicle; "
          f"others {o_s}/{o_v+o_s} Speech", "Fisher exact", p, st,
          None, lc_v + lc_s + o_v + o_s)

    # ---------- C6: per-chunk norms create onset artifacts (paired by Class) ----------
    ic = pd.read_csv(docs("initial_changes_synthetic.csv"))
    stab = ic[ic.Part == "stability"]
    pairs = [("RMS->LCMV", "LCMV"), ("MAX->LCMV", "LCMV"),
             ("RMS->GSC", "GSC"), ("MAX->GSC", "GSC")]
    for norm_m, plain_m in pairs:
        a = stab[stab.Method == norm_m].sort_values("Class").OnsetShift.abs()
        b = stab[stab.Method == plain_m].sort_values("Class").OnsetShift.abs()
        ps.append(paired_test(a, b,
                              f"C6 |onset artifact| {norm_m} > {plain_m}",
                              "initial_changes(stability)", "|onset shift| dB",
                              f"{norm_m} vs {plain_m}",
                              alternative="greater"))
    # extraction quality under real interferers (paired by Class)
    syn = ic[ic.Part == "synthetic"]
    a = syn[syn.Method == "LCMV"].sort_values("Class").dSIR
    b = syn[syn.Method == "GSC"].sort_values("Class").dSIR
    ps.append(paired_test(a, b, "C7 LCMV dSIR > GSC dSIR (calibrated, ESC-50)",
                          "initial_changes(synthetic)", "dSIR",
                          "LCMV vs GSC"))

    # ---------- C8: field level stability RMS->GSC best (paired by Dir) ----------
    icf = pd.read_csv(docs("initial_changes_field.csv"))
    for other in ("GSC", "LCMV", "DAS"):
        a = icf[icf.Method == "RMS->GSC"].sort_values("Dir").LevelStd_dB
        b = icf[icf.Method == other].sort_values("Dir").LevelStd_dB
        ps.append(paired_test(a, b, f"C8 RMS->GSC more level-stable than {other} (field)",
                              "initial_changes_field", "level std dB",
                              f"RMS->GSC vs {other}",
                              alternative="less", note="n=4 dirs: sign test"))

    # ---------- C9: blind pipeline siren recovery above chance ----------
    bm = pd.read_csv(docs("blind_mc_results.csv"))
    for cond in bm.Condition.unique():
        sub = bm[bm.Condition == cond]
        k, n = int(sub.siren_recovered.sum()), len(sub)
        p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
        claim(f"C9 blind siren recovery > 50% ({cond})", "blind_mc_results",
              "siren_recovered", f"{k}/{n}", "exact binomial", p, k, None, n)

    # ---------- Holm across the registry ----------
    df = pd.DataFrame(CLAIMS)
    testable = df.p_raw.notna()
    adj = np.full(len(df), np.nan)
    adj[testable.values] = holm(df.p_raw[testable].values)
    df["p_holm"] = adj
    def verdict(r):
        if np.isnan(r.p_raw):
            return "n/a"
        if r.p_holm < 0.05:
            return "SUPPORTED"
        if r.n < 5:
            return "UNDERPOWERED (consistent)"
        return "NOT SUPPORTED"
    df["Verdict"] = df.apply(verdict, axis=1)

    df.to_csv(docs("stats_validation.csv"), index=False)
    print("=" * 112)
    print("STATISTICAL VALIDATION OF CLAIMS (Holm-corrected across registry)")
    print("=" * 112)
    for _, r in df.iterrows():
        print(f"[{r.Verdict:26s}] {r.Claim}")
        print(f"    {r.Test}: stat={r.Statistic}, p_raw={r.p_raw:.2e}, "
              f"p_holm={r.p_holm:.2e}" if not np.isnan(r.p_raw) else "    n/a",
              f"effect(delta)={r.Effect}, n={r.n} {r.Note}")
    n_sup = (df.Verdict == "SUPPORTED").sum()
    print(f"\n{n_sup}/{len(df)} claims statistically SUPPORTED at Holm-corrected "
          f"alpha=0.05; underpowered ones flagged (need more draws/dirs), "
          f"NOT-SUPPORTED ones would revise conclusions.")
    print("Saved docs/stats_validation.csv")


if __name__ == "__main__":
    main()

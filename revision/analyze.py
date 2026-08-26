"""Aggregate the raw per-run CSVs into reportable tables with proper statistics.

Every comparison reports mean, sd, 95% CI, Welch t-test, Mann-Whitney U,
Cohen's d and Cliff's delta. Per-seed rows stay in the raw files.

Usage: python analyze.py [E1 E4 E10 E11 E3 E12 E2 E5 E7 COST]
"""
import os, math, sys
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 250)
DET = ("FIFO", "Consolidation", "Greedy(EDF+greenest)")


def label(r):
    m, i = r["method"], r["init"]
    if m in DET or m == "GA_OriginalGA":
        return m
    if i == "carbon":
        return "CA-WOA" if m == "WOA" else "CA-" + m
    if i == "improved":
        return m + "+improved"
    return m


def ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else np.nan, np.nan, np.nan)
    m, s = a.mean(), a.std(ddof=1)
    h = stats.t.ppf(0.975, len(a) - 1) * s / math.sqrt(len(a))
    return m, m - h, m + h


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b))


def cohend(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def compare(a, b, na, nb, **extra):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    try:
        _, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        pu = 1.0
    ma, la, ha = ci95(a); mb, lb, hb = ci95(b)
    return {**extra, "A": na, "B": nb, "n": len(a),
            "mean_A": round(ma, 3), "ci_A": "[%.2f, %.2f]" % (la, ha),
            "mean_B": round(mb, 3), "ci_B": "[%.2f, %.2f]" % (lb, hb),
            "diff_pp": round(ma - mb, 3), "welch_p": "%.3g" % pt, "mwu_p": "%.3g" % pu,
            "cohen_d": round(cohend(a, b), 2), "cliffs_delta": round(cliffs_delta(a, b), 3),
            "sig": "YES" if (pt < 0.05 and pu < 0.05) else
                   ("mixed" if (pt < 0.05) != (pu < 0.05) else "no")}


def load(exp):
    p = os.path.join(OUT, "raw_%s.csv" % exp)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["label"] = d.apply(label, axis=1)
    return d


def ranked(d, val="carbon_red_vs_naive_%", note=""):
    """Print a per-(N,M) ranking of stochastic methods with CIs."""
    st = d[~d.method.isin(DET)]
    for (n, m), grp in st.groupby(["N", "M"]):
        rows = []
        for lab, g in grp.groupby("label"):
            mu, lo, hi = ci95(g[val].values)
            rows.append((lab, mu, g[val].std(ddof=1), lo, hi, g["sla_%"].mean(),
                         g["overload_%"].mean()))
        rows.sort(key=lambda r: -r[1])
        print("\n  N=%d, M=%d%s" % (n, m, note))
        for i, (lab, mu, sd, lo, hi, sla, ov) in enumerate(rows, 1):
            mark = "  <-- CA-WOA" if lab == "CA-WOA" else ""
            print("    %d. %-14s %7.2f +/- %-6.3f [%6.2f, %6.2f]  SLA %5.2f%%  ovl %6.2f%%%s"
                  % (i, lab, mu, sd, lo, hi, sla, ov, mark))


def vs_greedy(d, tag):
    """The comparison Reviewer 2 asked for: CA-WOA vs the EDF+greenest heuristic."""
    print("\n  CA-WOA vs the deterministic greedy heuristic (%s):" % tag)
    print("  %-6s %-4s %-24s %-12s %-9s %s" % ("N", "M", "CA-WOA", "greedy", "diff pp", "p"))
    rows = []
    for (n, m), g in d.groupby(["N", "M"]):
        ca = g[(g.method == "WOA") & (g.init == "carbon")]["carbon_red_vs_naive_%"].values
        gr = g[g.method == "Greedy(EDF+greenest)"]["carbon_red_vs_naive_%"].values
        if not len(ca) or not len(gr):
            continue
        gv = float(gr[0])
        t, p = stats.ttest_1samp(ca, gv)
        rows.append({"N": n, "M": m, "ca_woa": round(ca.mean(), 3),
                     "ca_sd": round(ca.std(ddof=1), 3), "greedy": round(gv, 3),
                     "diff_pp": round(ca.mean() - gv, 3), "p": "%.3g" % p,
                     "winner": "CA-WOA" if ca.mean() > gv else "greedy"})
        print("  %-6d %-4d %7.3f +/- %-6.3f      %8.3f     %+8.3f  %-10s -> %s"
              % (n, m, ca.mean(), ca.std(ddof=1), gv, ca.mean() - gv, "%.3g" % p,
                 rows[-1]["winner"]))
    return pd.DataFrame(rows)


def sig_vs_cawoa(d, **extra):
    rows = []
    st = d[~d.method.isin(DET)]
    for (n, m), grp in st.groupby(["N", "M"]):
        by = {k: v["carbon_red_vs_naive_%"].values for k, v in grp.groupby("label")}
        if "CA-WOA" not in by:
            continue
        for o in sorted(k for k in by if k != "CA-WOA"):
            rows.append(compare(by["CA-WOA"], by[o], "CA-WOA", o, N=n, M=m, **extra))
    return pd.DataFrame(rows)


def header(t):
    print("\n" + "=" * 118); print(t); print("=" * 118)


def do_grid(exp, title):
    d = load(exp)
    if d is None:
        print("\n%s not ready" % exp); return
    header("%s  %s  (30 seeds)" % (exp, title))
    det = d[d.method.isin(DET)]
    dec = det.pivot_table(index=["N", "M"], columns="label",
                          values="carbon_red_vs_naive_%").round(2)
    if {"Consolidation", "Greedy(EDF+greenest)"} <= set(dec.columns):
        dec["consol_share_%"] = (dec["Consolidation"] / dec["Greedy(EDF+greenest)"] * 100).round(1)
        dec["timing_adds_pp"] = (dec["Greedy(EDF+greenest)"] - dec["Consolidation"]).round(2)
    print("\nDeterministic baselines and the consolidation / timing decomposition:")
    print(dec.to_string())
    print("\nCapacity pressure:")
    print(d.groupby(["N", "M"])[["peak_demand", "total_load", "arrival_slots",
                                 "overload_%"]].mean().round(2).to_string())
    ranked(d)
    g = vs_greedy(d, title)
    g.to_csv(os.path.join(OUT, "%s_vs_greedy.csv" % exp), index=False)
    s = sig_vs_cawoa(d)
    s.to_csv(os.path.join(OUT, "%s_stats.csv" % exp), index=False)
    wins = (s.assign(w=s.diff_pp > 0).groupby(["N", "M"])["w"].all())
    print("\n  CA-WOA beats every other metaheuristic in %d of %d (N,M) cells"
          % (int(wins.sum()), len(wins)))
    print("  Saved -> %s_stats.csv, %s_vs_greedy.csv" % (exp, exp))


def do_ablation(exp, title):
    d = load(exp)
    if d is None:
        print("\n%s not ready" % exp); return
    header("%s  ABLATION  %s  (30 seeds)" % (exp, title))

    def arm(r):
        if r["method"] == "Greedy(EDF+greenest)":
            return "greedy only (no search)"
        if r["init"] == "carbon" and r["beta"] == 0:
            return "CA-WOA, beta=0 (no SLA term)"
        if r["init"] == "carbon" and r["gamma"] == 0:
            return "CA-WOA, gamma=0 (no penalty term)"
        return label(r)
    d["arm"] = d.apply(arm, axis=1)
    woa = d[d.method.isin(["WOA", "Greedy(EDF+greenest)"])]
    print("\nWOA initialisation arms:")
    for (n, m), grp in woa.groupby(["N", "M"]):
        print("\n  N=%d, M=%d" % (n, m))
        rows = []
        for a, g in grp.groupby("arm"):
            mu, lo, hi = ci95(g["carbon_red_vs_naive_%"].values)
            rows.append((a, mu, g["carbon_red_vs_naive_%"].std(ddof=1), lo, hi))
        for a, mu, sd, lo, hi in sorted(rows, key=lambda r: -r[1]):
            sd_s = "%.3f" % sd if sd == sd else "  -  "
            print("    %-34s %7.2f +/- %-6s [%6.2f, %6.2f]" % (a, mu, sd_s, lo, hi))
    rows = []
    for (n, m), grp in woa.groupby(["N", "M"]):
        by = {k: v["carbon_red_vs_naive_%"].values for k, v in grp.groupby("arm")}
        for a, b in (("CA-WOA", "WOA"), ("CA-WOA", "WOA+improved"),
                     ("WOA+improved", "WOA"),
                     ("CA-WOA", "CA-WOA, beta=0 (no SLA term)"),
                     ("CA-WOA", "CA-WOA, gamma=0 (no penalty term)")):
            if a in by and b in by and len(by[a]) > 1 and len(by[b]) > 1:
                rows.append(compare(by[a], by[b], a, b, N=n, M=m))
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(OUT, "%s_ablation_stats.csv" % exp), index=False)
    print("\nAblation significance:")
    print(ab[["N", "M", "A", "B", "mean_A", "mean_B", "diff_pp", "welch_p",
              "mwu_p", "cohen_d", "sig"]].to_string(index=False))

    print("\n" + "-" * 118)
    print("Does carbon-aware seeding GENERALISE across optimisers?")
    print("-" * 118)
    gen = d[d.init.isin(["random", "carbon"]) & (d.beta == 0.3) & (d.gamma == 0.3)]
    rows = []
    for (n, m, alg), grp in gen.groupby(["N", "M", "method"]):
        if alg in DET:
            continue
        r_ = grp[grp.init == "random"]["carbon_red_vs_naive_%"].values
        c_ = grp[grp.init == "carbon"]["carbon_red_vs_naive_%"].values
        if len(r_) > 1 and len(c_) > 1:
            rows.append(compare(c_, r_, alg + "+carbon", alg + "+random",
                                N=n, M=m, algo=alg))
    gg = pd.DataFrame(rows)
    gg.to_csv(os.path.join(OUT, "%s_seeding_generalisation.csv" % exp), index=False)
    print(gg[["N", "M", "algo", "mean_A", "mean_B", "diff_pp", "welch_p", "mwu_p",
              "cohen_d", "sig"]].to_string(index=False))
    print("\n  seeding helps in %d of %d (algo,N,M) cells; mean gain %+.2f pp; "
          "significant in %d" % ((gg.diff_pp > 0).sum(), len(gg), gg.diff_pp.mean(),
                                 (gg.sig == "YES").sum()))


def do_E2():
    d = load("E2")
    if d is None:
        print("\nE2 not ready"); return
    header("E2  POWER MODELS (linear / cubic / piecewise), 30 seeds")
    det = d[d.method.isin(DET)]
    print("\nConsolidation and greedy benefit by power model:")
    print(det.pivot_table(index=["N", "M"], columns=["power", "label"],
                          values="carbon_red_vs_naive_%").round(2).to_string())
    st = d[~d.method.isin(DET)]
    print("\nCarbon reduction by method and power model (mean):")
    print(st.pivot_table(index="label", columns=["power", "N"],
                         values="carbon_red_vs_naive_%").round(2).to_string())
    print("\nStandard deviations:")
    print(st.pivot_table(index="label", columns=["power", "N"],
                         values="carbon_red_vs_naive_%", aggfunc="std").round(3).to_string())
    rows = []
    for (n, m, pw), grp in st.groupby(["N", "M", "power"]):
        by = {k: v["carbon_red_vs_naive_%"].values for k, v in grp.groupby("label")}
        if "CA-WOA" in by:
            for o in sorted(k for k in by if k != "CA-WOA"):
                rows.append(compare(by["CA-WOA"], by[o], "CA-WOA", o, N=n, M=m, power=pw))
    s = pd.DataFrame(rows); s.to_csv(os.path.join(OUT, "E2_stats.csv"), index=False)
    for pw, g in s.groupby("power"):
        w = g.assign(w=g.diff_pp > 0).groupby(["N", "M"])["w"].all()
        print("  %-10s CA-WOA best in %d of %d cells" % (pw, int(w.sum()), len(w)))


def do_E5():
    d = load("E5")
    if d is None:
        print("\nE5 not ready"); return
    header("E5  SENSITIVITY: fitness weights and seeded-population fraction")
    w = d[np.isclose(d.seed_frac, 1 / 3)].copy()
    w["abg"] = w.apply(lambda r: "(%.2f,%.2f,%.2f)" % (r.alpha, r.beta, r.gamma), axis=1)
    for cap in (False, True):
        for init in ("carbon", "random"):
            sub = w[(w.init == init) & (w.cap == cap)]
            if not len(sub):
                continue
            t = sub.pivot_table(index="abg", columns=["N", "M"],
                                values="carbon_red_vs_naive_%").round(3)
            nm = "CA-WOA" if init == "carbon" else "standard WOA"
            print("\n  %s, capacity-limited=%s -- carbon reduction by weight combination:"
                  % (nm, cap))
            print(t.to_string())
            print("  spread across the grid: " + ", ".join(
                "N=%d/M=%d: %.3f pp" % (c[0], c[1], t[c].max() - t[c].min())
                for c in t.columns))
            sl = sub.pivot_table(index="abg", columns=["N", "M"], values="sla_%")
            print("  SLA%% across the grid: %.3f to %.3f" % (sl.values.min(), sl.values.max()))
    print("\n  Seeded-population fraction sweep (carbon reduction):")
    f = (d[d.init == "carbon"].groupby(["cap", "N", "M", "seed_frac"])
         ["carbon_red_vs_naive_%"].agg(mean="mean", sd="std").round(3).reset_index())
    print(f.pivot_table(index="seed_frac", columns=["cap", "N", "M"],
                        values="mean").round(3).to_string())
    f.to_csv(os.path.join(OUT, "E5_seedfrac.csv"), index=False)


def do_E7():
    d = load("E7")
    if d is None:
        print("\nE7 not ready"); return
    header("E7  MULTIPLE WORKLOAD SUBSETS + NASA REAL ARRIVAL PROCESS")
    gg = d[d.wl == "google"]
    print("\nGoogle subsets (does the result depend on which sample is drawn?):")
    print(gg.pivot_table(index=["N", "M"], columns=["label", "subset"],
                         values="carbon_red_vs_naive_%").round(2).to_string())
    sp = (gg[gg.label == "CA-WOA"].groupby(["N", "M"])["carbon_red_vs_naive_%"]
          .agg(["mean", "std", "min", "max"]).round(3))
    print("\nCA-WOA across subsets:")
    print(sp.to_string())
    na = d[d.wl == "nasa"]
    if len(na):
        print("\nNASA-iPSC real arrival process:")
        ranked(na, note=" [NASA]")


def do_COST():
    frames = [load(e) for e in ("E1", "E4", "E10", "E11", "E2", "E3", "E12")]
    frames = [f for f in frames if f is not None]
    if not frames:
        print("\nCOST not ready"); return
    d = pd.concat(frames, ignore_index=True)
    d = d[d.nfe > 0]
    header("COMPUTATIONAL COST: objective evaluations, runtime, peak memory")
    print("\nObjective-function evaluations per run (the paper claims equal budgets):")
    print(d.groupby("method")["nfe"].agg(["mean", "min", "max"]).round(0).to_string())
    print("\nRuntime (s) by method and task count:")
    print(d.pivot_table(index="method", columns="N", values="runtime_s",
                        aggfunc="mean").round(3).to_string())
    print("\nPeak Python-allocated memory (MB):")
    print(d.pivot_table(index="method", columns="N", values="peak_mem_mb",
                        aggfunc="mean").round(1).to_string())
    g = load("GABUG")
    if g is not None:
        print("\nGA.OriginalGA defect, measured at scale:")
        print(g.groupby(["method", "N"])[["nfe", "carbon_red_vs_naive_%", "runtime_s"]]
              .mean().round(2).to_string())


def do_feas(exp, title):
    """Capacity treated as a FEASIBILITY constraint: energy is charged on full
    demand and overload is penalised, so overloading can no longer be profitable.
    The question is not 'who is greenest' but 'who produces a schedule that fits'."""
    d = load(exp)
    if d is None:
        print("\n%s not ready" % exp); return
    header("%s  %s  (30 seeds)" % (exp, title))

    print("\nFeasibility rate (fraction of runs with zero overload):")
    fr = d.pivot_table(index="label", columns=["N", "M"], values="feasible",
                       aggfunc="mean").round(2)
    print(fr.to_string())

    print("\nMean overload %% of demanded load (0 = feasible):")
    print(d.pivot_table(index="label", columns=["N", "M"], values="overload_%",
                        aggfunc="mean").round(2).to_string())

    print("\nCarbon reduction vs naive FIFO, ALL runs (infeasible ones are not comparable):")
    print(d.pivot_table(index="label", columns=["N", "M"],
                        values="carbon_red_vs_naive_%", aggfunc="mean").round(2).to_string())

    f = d[d.feasible]
    print("\nAmong FEASIBLE runs only -- carbon reduction, mean +- sd (n runs):")
    if f.empty:
        print("  none")
    else:
        agg = (f.groupby(["N", "M", "label"])["carbon_red_vs_naive_%"]
               .agg(["mean", "std", "size"]).round(2))
        print(agg.to_string())

    print("\nSLA violations %% (hard constraint should hold these at zero):")
    print(d.pivot_table(index="label", columns=["N", "M"], values="sla_%",
                        aggfunc="max").round(4).to_string())

    ranked(f if not f.empty else d)
    s2 = sig_vs_cawoa(d)
    s2.to_csv(os.path.join(OUT, "%s_stats.csv" % exp), index=False)
    fr.to_csv(os.path.join(OUT, "%s_feasibility.csv" % exp))
    print("\n  Saved -> %s_stats.csv, %s_feasibility.csv" % (exp, exp))


FNS = {"E1": lambda: do_grid("E1", "SCALABILITY, published model, soft penalty"),
       "E4": lambda: do_grid("E4", "HARD deadline constraint (repair)"),
       "E10": lambda: do_grid("E10", "CAPACITY-LIMITED (M is a real host limit)"),
       "E11": lambda: do_grid("E11", "CAPACITY-LIMITED + hard deadline constraint"),
       "E3": lambda: do_ablation("E3", "(published model)"),
       "E12": lambda: do_ablation("E12", "(capacity-limited)"),
       "E20": lambda: do_feas("E20", "CAPACITY FEASIBILITY + hard deadlines"),
       "E21": lambda: do_feas("E21", "CAPACITY FEASIBILITY, soft deadlines"),
       "E22": lambda: do_ablation("E22", "(under capacity feasibility constraint)"),
       "E2": do_E2, "E5": do_E5, "E7": do_E7, "COST": do_COST}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(FNS)):
        FNS[w]()

"""Regenerate every figure in main.tex from the committed raw results.

Addresses Reviewer 2 s5: axis labels with units, legends inside the axes, error
information on every mean (95% confidence intervals over 30 seeds), tick labels
readable at print size, and CA-WOA distinguished by marker and hatch as well as
colour so the figures survive greyscale printing.

Run:  python make_figures.py
Writes: ../../carbon-aware-scheduler-updated/figures/*.png  (300 dpi)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..",
                                  "carbon-aware-scheduler-updated", "figures"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

HL = "#c1272d"          # CA-WOA highlight
ORDER = ["CA-WOA", "WOA", "GWO", "PSO", "DE", "HHO", "GA"]
MARK = {"CA-WOA": "o", "WOA": "s", "GWO": "^", "PSO": "v",
        "DE": "D", "HHO": "P", "GA": "X"}
# Explicit palette: matplotlib's default cycle assigns a red close to the CA-WOA
# highlight, which made two series indistinguishable in the legend. None of these
# collide with HL, and every series also has its own marker for greyscale printing.
COL = {"CA-WOA": HL, "WOA": "#1f77b4", "GWO": "#ff7f0e", "PSO": "#2ca02c",
       "DE": "#9467bd", "HHO": "#8c564b", "GA": "#7f7f7f"}
col = lambda l: COL.get(l, "#4a6f8a")


def load(exp):
    d = pd.read_csv(os.path.join(HERE, "raw_%s.csv" % exp))
    d["label"] = np.where((d.method == "WOA") & (d.init == "carbon"),
                          "CA-WOA", d.method)
    return d


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote %s" % p)


# ------------------------------------------------------------------ figure 1
def fig_temporal():
    """Algorithm comparison: carbon vs SLA, with 95% CIs. The point is that the
    methods which look competitive on carbon are the ones missing deadlines."""
    d = load("E1"); d = d[(d.nfe > 0) & (d.N == 3000) & (d.M == 10)]
    labs = [l for l in ORDER if l in set(d.label)]
    car = [d[d.label == l]["carbon_red_vs_naive_%"] for l in labs]
    sla = [d[d.label == l]["sla_%"] for l in labs]
    x = np.arange(len(labs))
    cols = [HL if l == "CA-WOA" else "#4a6f8a" for l in labs]
    hat = ["///" if l == "CA-WOA" else "" for l in labs]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].bar(x, [c.mean() for c in car], yerr=[ci95(c) for c in car],
              capsize=4, color=cols, hatch=hat, edgecolor="black", linewidth=0.6)
    ax[0].set_ylabel("Carbon reduction (%)")
    ax[0].set_ylim(85, 89)
    ax[0].set_title("(a) Carbon reduction vs naive FIFO")

    ax[1].bar(x, [s.mean() for s in sla], yerr=[ci95(s) for s in sla],
              capsize=4, color=cols, hatch=hat, edgecolor="black", linewidth=0.6)
    ax[1].set_ylabel("SLA violations (% of tasks)")
    ax[1].set_title("(b) Deadline violations")
    ax[1].axhline(0, color="black", linewidth=0.8)

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labs, rotation=20)
        a.set_xlabel("Method")
    fig.suptitle("Published model, 3000 tasks on 10 hosts, 30 seeds "
                 "(error bars: 95% confidence interval)", y=1.02, fontsize=10)
    save(fig, "fig_temporal_comparison.png")


# ------------------------------------------------------------------ figure 2
def fig_capacity():
    """Feasibility first: the heuristics overload as load rises, the search
    methods do not. Carbon is only comparable among feasible schedules."""
    d = load("E20")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    heur = ["FIFO", "Consolidation", "Greedy(EDF+greenest)",
            "Threshold(WaitAwhile)"]
    styles = {"FIFO": ("--", "s"), "Consolidation": (":", "^"),
              "Greedy(EDF+greenest)": ("-", "D")}
    for m in heur:
        g = d[(d.label == m) & (d.M == 10)].groupby("N")["overload_%"].mean()
        if g.empty:
            continue
        ls, mk = styles.get(m, ("-.", "v"))
        ax[0].plot(g.index, g.values, ls, marker=mk, label=m.replace("(EDF+greenest)", " (EDF+greenest)"))
    ca = d[(d.label == "CA-WOA") & (d.M == 10)].groupby("N")["overload_%"].mean()
    ax[0].plot(ca.index, ca.values, "-", marker="o", color=HL, linewidth=2.4,
               label="CA-WOA (and all metaheuristics)")
    ax[0].axhline(0, color="black", linewidth=0.8)
    ax[0].set_xlabel("Task count $N$"); ax[0].set_ylabel("Overload (% of demanded load)")
    ax[0].set_title("(a) Feasibility: any value > 0 is infeasible")
    ax[0].legend(loc="upper left", framealpha=0.9)

    f = d[(d.feasible) & (d.nfe > 0) & (d.M == 10)]
    for l in [x for x in ORDER if x in set(f.label)]:
        g = f[f.label == l].groupby("N")["carbon_red_vs_naive_%"]
        m, e = g.mean(), g.apply(ci95)
        ax[1].errorbar(m.index, m.values, yerr=e.values, capsize=3,
                       marker=MARK[l], linewidth=2.4 if l == "CA-WOA" else 1.2,
                       color=col(l),
                       zorder=5 if l == "CA-WOA" else 2, label=l)
    ax[1].set_xlabel("Task count $N$"); ax[1].set_ylabel("Carbon reduction (%)")
    ax[1].set_title("(b) Carbon among feasible schedules only")
    ax[1].legend(loc="lower right", ncol=2, framealpha=0.9)
    fig.suptitle("Capacity as a feasibility constraint, 10 hosts, 30 seeds "
                 "(error bars: 95% confidence interval)", y=1.02, fontsize=10)
    save(fig, "fig_capacity_comparison.png")


# ------------------------------------------------------------------ figure 3
def fig_scalability():
    """500-3000 tasks at both host counts, with confidence intervals."""
    d = load("E1"); d = d[d.nfe > 0]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for j, M in enumerate((10, 20)):
        s = d[d.M == M]
        for l in [x for x in ORDER if x in set(s.label)]:
            g = s[s.label == l].groupby("N")
            for row, metric in ((0, "carbon_red_vs_naive_%"), (1, "sla_%")):
                m, e = g[metric].mean(), g[metric].apply(ci95)
                ax[row][j].errorbar(m.index, m.values, yerr=e.values, capsize=3,
                                    marker=MARK[l],
                                    linewidth=2.4 if l == "CA-WOA" else 1.2,
                                    color=col(l),
                                    zorder=5 if l == "CA-WOA" else 2, label=l)
        ax[0][j].set_title("$M=%d$ hosts" % M)
        ax[0][j].set_ylabel("Carbon reduction (%)")
        ax[1][j].set_ylabel("SLA violations (% of tasks)")
        for row in (0, 1):
            ax[row][j].set_xlabel("Task count $N$")
    ax[0][0].legend(loc="lower right", ncol=2, framealpha=0.9)
    fig.suptitle("Scalability, 500-3000 tasks, 30 seeds "
                 "(error bars: 95% confidence interval)", y=0.995, fontsize=10)
    fig.tight_layout()
    save(fig, "fig_scalability.png")


# ------------------------------------------------------------------ figure 4
def fig_forecast():
    """MAE by horizon with standard deviations over 5 repeats. The naive baselines
    are already present in raw_F_accuracy.csv, so they are read from there only --
    plotting them from the separate summary file as well duplicated every series."""
    d = pd.read_csv(os.path.join(HERE, "raw_F_accuracy.csv"))
    NAIVE = {"Persistence", "SeasonalNaive-24h", "SeasonalNaive-1week"}
    STYLE = {                      # distinct colour AND marker for every series
        "Ensemble":            (HL,        "o", "-",  2.6),
        "GradBoost":           ("#1f77b4", "s", "-",  1.3),
        "GRU":                 ("#2ca02c", "^", "-",  1.3),
        "LSTM":                ("#9467bd", "v", "-",  1.3),
        "CNN-LSTM":            ("#ff7f0e", "D", "-",  1.3),
        "Persistence":         ("#7f7f7f", "x", "--", 1.1),
        "SeasonalNaive-24h":   ("#8c564b", "+", ":",  1.1),
        "SeasonalNaive-1week": ("#17becf", "1", ":",  1.1),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    g = d.groupby(["model", "horizon_h"])["MAE"]
    mean, sd = g.mean().unstack(0), g.std().unstack(0)
    order = [m for m in STYLE if m in mean.columns]
    for m in order:
        c, mk, ls, lw = STYLE[m]
        lab = m + (" (naive)" if m in NAIVE else "")
        ax[0].errorbar(mean.index, mean[m].values, yerr=sd[m].fillna(0).values,
                       capsize=3, marker=mk, linestyle=ls, color=c, linewidth=lw,
                       zorder=6 if m == "Ensemble" else 2, label=lab)
    ax[0].set_xlabel("Forecast horizon (h)")
    ax[0].set_ylabel("MAE (gCO$_2$/kWh)")
    ax[0].set_title("(a) Accuracy vs horizon")
    ax[0].set_ylim(0, 52)
    ax[0].legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.95)

    h12 = d[d.horizon_h == 12.0]
    mm = h12.groupby("model")["MAE"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False)
    y = np.arange(len(mm))
    ax[1].barh(y, mm["mean"].values, xerr=mm["std"].fillna(0).values, capsize=4,
               color=[HL if i == "Ensemble" else
                      ("#9aa5ad" if i in NAIVE else "#4a6f8a") for i in mm.index],
               hatch=["///" if i == "Ensemble" else "" for i in mm.index],
               edgecolor="black", linewidth=0.6)
    ax[1].set_yticks(y)
    SHORT = {"SeasonalNaive-24h": "Seasonal-24h", "SeasonalNaive-1week": "Seasonal-1wk"}
    ax[1].set_yticklabels([SHORT.get(i, i) for i in mm.index], fontsize=9.5)
    ax[1].set_xlabel("MAE (gCO$_2$/kWh)")
    ax[1].set_title("(b) 12-hour horizon (naive baselines in grey)")
    for i, v in enumerate(mm["mean"].values):
        ax[1].text(v + 0.6, i, "%.2f" % v, va="center", fontsize=8.5)
    ax[1].set_xlim(0, mm["mean"].max() * 1.18)
    fig.suptitle("Carbon-intensity forecasting: direct multi-step prediction, "
                 "mean of 5 repeats (error bars: 1 s.d.)", y=1.02, fontsize=10)
    save(fig, "fig_forecast_comparison.png")


# ------------------------------------------------------------------ figure 5
def fig_convergence():
    """Explicitly requested: convergence curves and the budget caveat."""
    d = load("CONV"); d = d[d.curve.notna() & (d.N == 3000) & (d.M == 10)]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for l in [x for x in ORDER if x in set(d.label)]:
        cs = np.array([[float(v) for v in c.split(";")]
                       for c in d[d.label == l].curve])
        m, sd = cs.mean(0), cs.std(0)
        ep = np.arange(1, len(m) + 1)
        ax.plot(ep, m, linewidth=2.4 if l == "CA-WOA" else 1.2,
                color=col(l),
                zorder=5 if l == "CA-WOA" else 2, label=l)
        ax.fill_between(ep, m - sd, m + sd, alpha=0.12,
                        color=col(l))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Best fitness (lower is better)")
    ax.set_yscale("log")
    ax.set_title("Convergence, 3000 tasks on 10 hosts, 10 seeds\n"
                 "(shaded: $\\pm$1 s.d.; GA and GWO have not converged at epoch 120)",
                 fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig_convergence.png")


# ------------------------------------------------------------------ figure 6
def fig_regions():
    """The boundary result: the benefit of seeding tracks how much the grid's
    carbon intensity actually varies."""
    import core as K
    from scipy import stats
    a = load("E26")
    try:
        a = pd.concat([a, load("E28")], ignore_index=True)
    except FileNotFoundError:
        pass
    rows = []
    for r, g in a.groupby("region"):
        ca = g[g.label == "CA-WOA"]["carbon_red_vs_naive_%"]
        wo = g[g.label == "WOA"]["carbon_red_vs_naive_%"]
        if len(ca) < 5 or len(wo) < 5:
            continue
        allv = np.concatenate(K.carbon_windows(r))
        rows.append({"region": r, "var": allv.max() / allv.min(),
                     "gain": ca.mean() - wo.mean(), "ci": ci95(ca) + ci95(wo),
                     "sig": stats.mannwhitneyu(ca, wo).pvalue < 0.05})
    t = pd.DataFrame(rows).sort_values("var")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(t["var"], t["gain"], yerr=t["ci"], fmt="o", capsize=4,
                color=HL, markersize=9, linewidth=1.4)
    for _, r in t.iterrows():
        ax.annotate(r["region"], (r["var"], r["gain"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.axhline(0, color="black", linewidth=0.9)
    lo = t.iloc[0]
    ax.annotate("boundary case:\nnearly flat grid,\nCA-WOA loses to the\nbest method here",
                (lo["var"], lo["gain"]), textcoords="offset points",
                xytext=(34, 26), fontsize=8.5, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.set_xlabel("Grid carbon variability (max / min intensity over 111 days)")
    ax.set_ylabel("Seeding gain: CA-WOA $-$ WOA (percentage points)")
    ax.set_title("Seeding gain by grid carbon variability\n"
                 "1500 tasks, 6 real windows per region (95% confidence interval). "
                 "The\nleast variable grid gains least, but variability alone does not "
                 "rank the rest.", fontsize=9)
    save(fig, "fig_regions.png")


if __name__ == "__main__":
    print("regenerating figures from committed raw results:")
    for f in (fig_temporal, fig_capacity, fig_scalability,
              fig_forecast, fig_convergence, fig_regions):
        try:
            f()
        except Exception as e:
            print("  FAILED %s: %s: %s" % (f.__name__, type(e).__name__, e))

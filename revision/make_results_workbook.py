"""Build a single Excel workbook of the headline results.

The raw evidence is 52 CSV files with one row per (configuration, method, seed).
That is the right format for re-analysis but a poor one for review, so this script
consolidates the tables that appear in the manuscript into one workbook, one sheet
per table, plus an index and the full per-seed data for the main experiments.

Run:  python make_results_workbook.py
Out:  ../results/CA-WOA_results.xlsx
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "CA-WOA_results.xlsx")


def load(exp):
    d = pd.read_csv(os.path.join(HERE, "raw_%s.csv" % exp))
    d["label"] = np.where((d.method == "WOA") & (d.init == "carbon"), "CA-WOA", d.method)
    return d


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


sheets = {}

# ---- index -------------------------------------------------------------------
sheets["00_Index"] = pd.DataFrame([
    ("01_Algorithm_comparison", "3000 tasks, 10 hosts, 30 seeds: carbon and SLA per method"),
    ("02_Feasibility", "Overload by method and task count; any non-zero value is infeasible"),
    ("03_Scalability", "500-3000 tasks at 10 and 20 hosts, mean with 95% confidence interval"),
    ("04_Ablation", "Contribution of seeding and of each fitness term"),
    ("05_Weight_sensitivity", "Eleven (alpha,beta,gamma) combinations with the constraints they break"),
    ("06_Power_models", "Linear, cubic and piecewise power models"),
    ("07_Min_active_host", "Warm-pool constraint: CA-WOA against greedy and against plain WOA"),
    ("08_Heterogeneous", "Three fleets x three startup-overhead levels"),
    ("09_Carbon_regions", "Five real grid regions and their variability"),
    ("10_Workload_subsets", "Four disjoint Google subsets and the NASA-iPSC trace"),
    ("11_Cost_convergence", "Runtime, memory, evaluation counts, convergence epochs"),
    ("12_Forecast_accuracy", "MAE/RMSE/MAPE/MASE by horizon against naive baselines"),
    ("13_Forecast_coupling", "Predicted vs reactive vs perfect foresight at equal budget"),
    ("14_Forecast_degradation", "Carbon loss as forecast error increases"),
    ("15_GA_defect", "GA.OriginalGA performs 40 evaluations, not 4,840"),
    ("RAW_E1_published", "Full per-seed data, published model"),
    ("RAW_E20_feasibility", "Full per-seed data, capacity as a feasibility constraint"),
], columns=["Sheet", "Contents"])

# ---- 01 algorithm comparison -------------------------------------------------
d = load("E1"); d = d[(d.nfe > 0) & (d.N == 3000) & (d.M == 10)]
t = d.groupby("label").agg(
    carbon_reduction_pct=("carbon_red_vs_naive_%", "mean"),
    carbon_sd=("carbon_red_vs_naive_%", "std"),
    sla_violations_pct=("sla_%", "mean"),
    sla_sd=("sla_%", "std"),
    evaluations=("nfe", "mean")).round(3).sort_values("carbon_reduction_pct", ascending=False)
sheets["01_Algorithm_comparison"] = t.reset_index()

# ---- 02 feasibility ----------------------------------------------------------
d = load("E20")
sheets["02_Feasibility"] = (d[d.M == 10].pivot_table(index="label", columns="N",
                            values="overload_%").round(2).reset_index())

# ---- 03 scalability ----------------------------------------------------------
d = load("E1"); d = d[d.nfe > 0]
g = d.groupby(["M", "N", "label"])["carbon_red_vs_naive_%"]
sc = pd.DataFrame({"mean": g.mean(), "ci95": g.apply(ci95), "sd": g.std()}).round(3)
sheets["03_Scalability"] = sc.reset_index()

# ---- 04 ablation -------------------------------------------------------------
d = load("E3"); d = d[(d.N == 3000) & (d.M == 10)]
def arm(r):
    if r["method"] == "Greedy(EDF+greenest)": return "greedy only (no search)"
    if r["method"] != "WOA": return None
    if r["init"] == "carbon" and r["beta"] == 0:  return "CA-WOA, beta=0 (no SLA term)"
    if r["init"] == "carbon" and r["gamma"] == 0: return "CA-WOA, gamma=0 (no overload term)"
    return {"random": "standard WOA", "improved": "WOA + improved init (no carbon)",
            "carbon": "CA-WOA"}[r["init"]]
d = d.assign(arm=d.apply(arm, axis=1)); d = d[d.arm.notna()]
sheets["04_Ablation"] = (d.groupby("arm")[["carbon_red_vs_naive_%", "sla_%", "overload_%"]]
                         .mean().round(3).reset_index())

# ---- 05 weights --------------------------------------------------------------
d = load("E5")
d["weights"] = d.apply(lambda r: "(%.2f,%.2f,%.2f)" % (r.alpha, r.beta, r.gamma), axis=1)
w = d[(d.init == "carbon") & (~d.cap) & (d.seed_frac.round(4) == 0.3333)]
sheets["05_Weight_sensitivity"] = (w.groupby("weights")[
    ["carbon_red_vs_naive_%", "sla_%", "overload_%"]].mean().round(3)
    .sort_values("carbon_red_vs_naive_%", ascending=False).reset_index())

# ---- 06 power models ---------------------------------------------------------
d = load("E2"); d = d[d.nfe > 0]
sheets["06_Power_models"] = (d.pivot_table(index="label", columns=["power", "N"],
                             values="carbon_red_vs_naive_%").round(3).reset_index())

# ---- 07 minimum active host --------------------------------------------------
d = load("E23"); rows = []
for mm in ["0", "0.25", "0.5", "auto"]:
    s = d[d.mmin.astype(str) == mm]; vg, vw = [], []
    for (N, M), g2 in s.groupby(["N", "M"]):
        ca = g2[g2.label == "CA-WOA"]["carbon_red_vs_naive_%"]
        gr = g2[g2.label == "Greedy(EDF+greenest)"]["carbon_red_vs_naive_%"]
        wo = g2[g2.label == "WOA"]["carbon_red_vs_naive_%"]
        if len(gr): vg.append(ca.mean() - gr.mean())
        if len(wo) > 3: vw.append(ca.mean() - wo.mean())
    rows.append({"min_active_hosts": mm,
                 "beats_greedy": "%d/%d" % (sum(v > 0 for v in vg), len(vg)),
                 "vs_greedy_points": round(np.mean(vg), 3),
                 "vs_plain_WOA_points": round(np.mean(vw), 3)})
sheets["07_Min_active_host"] = pd.DataFrame(rows)

# ---- 08 heterogeneous fleets -------------------------------------------------
d = load("E29")
sheets["08_Heterogeneous"] = (d[d.N == 1500].pivot_table(index="label",
    columns=["fleet", "startup"], values="carbon_red_vs_naive_%").round(3).reset_index())

# ---- 09 carbon regions -------------------------------------------------------
import core as K
rows = []
for r in ("UK", "IE", "NI", "US-CAL", "US-MIDA"):
    f = os.path.join(HERE, "..", "data", "carbon",
                     "carbon_history%s.csv" % ("" if r == "UK" else "_" + r))
    dd = pd.read_csv(f)
    rows.append({"region": r, "slots": len(dd), "mean_gCO2_per_kWh": round(dd.intensity.mean(), 1),
                 "min": round(dd.intensity.min()), "max": round(dd.intensity.max()),
                 "variability_max_over_min": round(dd.intensity.max() / dd.intensity.min(), 2)})
sheets["09_Carbon_regions"] = pd.DataFrame(rows).sort_values(
    "variability_max_over_min", ascending=False)

# ---- 10 workload subsets -----------------------------------------------------
d = load("E7"); d = d[d.nfe > 0]
sheets["10_Workload_subsets"] = (d[d.wl == "nasa"].groupby("label")[
    ["carbon_red_vs_naive_%", "sla_%"]].mean().round(3)
    .sort_values("carbon_red_vs_naive_%", ascending=False).reset_index())

# ---- 11 cost and convergence -------------------------------------------------
a = pd.concat([load(e) for e in ("E1", "E2", "E3", "E4", "E20", "E21", "E22")],
              ignore_index=True)
a = a[a.nfe > 0]
cost = a.groupby("method").agg(mean_evaluations=("nfe", "mean"),
                               min_evaluations=("nfe", "min"),
                               max_evaluations=("nfe", "max"),
                               mean_runtime_s=("runtime_s", "mean"),
                               peak_memory_MB=("peak_mem_mb", "max")).round(2)
cv = load("CONV"); cv = cv[cv.curve.notna() & (cv.N == 3000) & (cv.M == 10)]
conv = {}
for l, g2 in cv.groupby("label"):
    m = np.array([[float(v) for v in c.split(";")] for c in g2.curve]).mean(0)
    conv[l] = (round(m[0], 5), round(m[119], 5), int(np.argmax(m <= m[-1] * 1.01)) + 1)
cost["epoch1_fitness"] = [conv.get(i, (None,) * 3)[0] for i in cost.index]
cost["final_fitness"] = [conv.get(i, (None,) * 3)[1] for i in cost.index]
cost["epochs_to_within_1pct"] = [conv.get(i, (None,) * 3)[2] for i in cost.index]
sheets["11_Cost_convergence"] = cost.reset_index()

# ---- 12-14 forecasting -------------------------------------------------------
fa = pd.read_csv(os.path.join(HERE, "raw_F_accuracy.csv"))
sheets["12_Forecast_accuracy"] = (fa.groupby(["horizon_h", "model"])[
    ["MAE", "RMSE", "MAPE_%", "MASE"]].agg(["mean", "std"]).round(3).reset_index())
fc = pd.read_csv(os.path.join(HERE, "F_coupling_summary.csv"))
fc = fc.copy()
sheets["13_Forecast_coupling"] = fc.round(4)
sheets["14_Forecast_degradation"] = pd.read_csv(
    os.path.join(HERE, "F_degradation_summary.csv")).round(3)

# ---- 15 GA defect ------------------------------------------------------------
gb = load("GABUG")
sheets["15_GA_defect"] = (gb.groupby(["method", "N"])[
    ["nfe", "carbon_red_vs_naive_%", "runtime_s"]].mean().round(3).reset_index())

# ---- raw per-seed data for the two main experiments --------------------------
sheets["RAW_E1_published"] = load("E1")
sheets["RAW_E20_feasibility"] = load("E20")

# ---- write -------------------------------------------------------------------
def flatten(df):
    """Excel cannot take MultiIndex columns without an index column, so join them."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(p) for p in c if str(p) != "").strip()
                      for c in df.columns.to_flat_index()]
    return df

with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
    for name, df in sheets.items():
        flatten(df).to_excel(xl, sheet_name=name[:31], index=False)
print("wrote %s" % os.path.abspath(OUT))
print("%d sheets:" % len(sheets))
for name, df in sheets.items():
    print("   %-26s %5d rows x %2d cols" % (name, len(df), len(df.columns)))

"""Simulating existing metaheuristic algorithms on a common benchmark (level playing field).
Runs GWO, PSO, DE, WOA, HHO, GA on the SAME real workload + real carbon data, measuring
energy, carbon, cost and SLA for each — vs a non-carbon-aware FIFO/Round-Robin baseline.
Real data: NASA-iPSC workload + UK National Grid carbon intensity. Platform: Python + Mealpy."""
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mealpy import FloatVar, GWO, PSO, DE, WOA, HHO, GA

ROOT = "/home/durga/carbon-aware-scheduler"
SEED = 42

# ---- energy model ----
P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5
def slot_energy_kwh(u): return (P_IDLE_W + (P_MAX_W - P_IDLE_W) * u) * SLOT_H / 1000.0

# ---- real carbon + price ----
j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI)//48 + 1):
    for s in range(32, 40):
        k = day*48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)

# ---- real workload ----
cols = ["job","submit","wait","runtime","nproc","avg_cpu","used_mem","req_proc","req_time",
        "req_mem","status","uid","gid","app","queue","partition","prev_job","think"]
rows = [ln.split()[:18] for ln in open(ROOT + "/data/workload/NASA.swf")
        if ln.strip() and not ln.strip().startswith(";") and len(ln.split()) >= 18]
raw = pd.DataFrame(rows, columns=cols).astype(float)
raw = raw[(raw.runtime > 0) & (raw.nproc > 0)].head(60).reset_index(drop=True)
maxp, smin, smax = raw.nproc.max(), raw.submit.min(), raw.submit.max()
SLACK, MAX_DEFER = 8, 24
tasks = []
for _, r in raw.iterrows():
    dur = int(np.clip(math.ceil(r.runtime/1800.0), 1, 12))
    u = float(np.clip(r.nproc/maxp, 0.05, 1.0))
    e = int((r.submit - smin)/(smax - smin + 1) * (H//3))
    tasks.append({"dur": dur, "u": u, "earliest": e, "deadline": e + dur + SLACK})

# ---- metrics + schedulers ----
def evaluate(starts):
    energy = carbon_g = cost = 0.0; viol = 0
    for t, s in zip(tasks, starts):
        e = slot_energy_kwh(t["u"]); run = [k for k in range(s, s+t["dur"]) if k < H]
        energy += e*len(run); carbon_g += sum(e*CI[k] for k in run); cost += sum(e*PRICE[k] for k in run)
        if s + t["dur"] > t["deadline"]: viol += 1
    return {"Energy_kWh": energy, "Carbon_kgCO2": carbon_g/1000.0, "Cost_GBP": cost,
            "SLA_viol_%": 100.0*viol/len(tasks)}

def decode(x):
    out = []
    for xi, t in zip(x, tasks):
        room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
        out.append(t["earliest"] + int(round(xi*room)))
    return out

base = evaluate([t["earliest"] for t in tasks])          # FIFO / Round-Robin baseline
def fitness(x):
    m = evaluate(decode(x))
    # carbon objective + strong SLA-deadline penalty (treat deadlines as near-hard constraints)
    return 1.0*(m["Carbon_kgCO2"]/base["Carbon_kgCO2"]) + 3.0*(m["SLA_viol_%"]/100.0)

# ---- run each existing metaheuristic on the SAME problem ----
ALGOS = {"GWO": GWO.OriginalGWO, "PSO": PSO.OriginalPSO, "DE": DE.OriginalDE,
         "WOA": WOA.OriginalWOA, "HHO": HHO.OriginalHHO, "GA": GA.OriginalGA}
results = {"FIFO/Round-Robin (baseline)": base}
for name, cls in ALGOS.items():
    try:
        problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*len(tasks), ub=[1.0]*len(tasks)),
                   "minmax": "min", "log_to": None}
        model = cls(epoch=150, pop_size=50)
        g = model.solve(problem, seed=SEED)
        results[name] = evaluate(decode(g.solution))
        print("ran %-4s OK" % name)
    except Exception as ex:
        print("ran %-4s FAILED: %s" % (name, ex))

# ---- comparison table ----
df = pd.DataFrame(results).T.round(3)
df["CarbonReduction_%"] = ((base["Carbon_kgCO2"] - df["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100).round(1)
df.loc["FIFO/Round-Robin (baseline)", "CarbonReduction_%"] = 0.0
print("\n" + "="*78)
print("SIMULATING EXISTING ALGORITHMS — common workload (60 NASA tasks) + real carbon data")
print("="*78)
print(df.to_string())
df.to_csv(ROOT + "/results/algorithm_comparison.csv")

# ---- chart ----
algo_rows = df.drop(index="FIFO/Round-Robin (baseline)")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(algo_rows.index, algo_rows["CarbonReduction_%"], color="tab:green")
ax[0].set(title="Carbon reduction vs baseline (%)", ylabel="%"); ax[0].tick_params(axis='x', rotation=0)
ax[1].bar(df.index, df["SLA_viol_%"], color="tab:orange")
ax[1].set(title="SLA violations (%)", ylabel="%"); ax[1].tick_params(axis='x', rotation=45)
fig.tight_layout(); fig.savefig(ROOT + "/results/algorithm_comparison.png", dpi=120)
print("\nSaved -> results/algorithm_comparison.csv and .png")

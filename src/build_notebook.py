"""Builds the local Jupyter notebook (VS Code): SIMULATING EXISTING ALGORITHMS on the REAL
Google Cluster Trace 2011, reading the downloaded chunk + real carbon data from disk."""
import nbformat as nbf

ROOT = "/home/durga/carbon-aware-scheduler"
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Energy-Efficient and Carbon-Aware Cloud Task Scheduling — Simulating Existing Algorithms
**Navaneetha Thalakokkula — MSc Cloud Computing, National College of Ireland**

Simulates existing metaheuristic algorithms (GWO, PSO, DE, WOA, HHO, GA) on a **common benchmark** —
the **real Google Cluster Trace 2011** and **real UK grid-carbon data** — comparing energy, carbon,
cost and SLA on a level playing field. Baseline = non-carbon-aware FIFO/Round-Robin.

**Platform:** Python · Mealpy · NumPy · pandas · Matplotlib.
**Real data:** Google Cluster Trace 2011 · UK National Grid ESO carbon intensity.""")

md("## 1. Setup")
code('''import json, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
%matplotlib inline
from mealpy import FloatVar, GWO, PSO, DE, WOA, HHO, GA
ROOT = "/home/durga/carbon-aware-scheduler"
SEED = 42; np.random.seed(SEED)
print("Mealpy ready.")''')

md(r"""## 2. Energy model
$$ P(u) = P_{idle} + (P_{max}-P_{idle})\,u, \qquad E = \sum_t P(u_t)\,\Delta t \;(\text{kWh}). $$""")
code('''P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5
def slot_energy_kwh(u):
    return (P_IDLE_W + (P_MAX_W - P_IDLE_W) * u) * SLOT_H / 1000.0''')

md(r"""## 3. Real carbon intensity and cost
$$ CO_2 = \sum_t E_t \cdot CI_t, \qquad Cost = \sum_t E_t \cdot price_t. $$""")
code('''with open(ROOT + "/data/carbon/3day_window.json") as f:
    j = json.load(f)
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI)//48 + 1):
    for s in range(32, 40):
        k = day*48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)
print("Carbon: %d slots, %.0f-%.0f gCO2/kWh" % (H, CI.min(), CI.max()))
plt.figure(figsize=(10, 3)); plt.plot(CI, color="tab:red")
plt.title("Real UK grid carbon intensity"); plt.xlabel("half-hour slot"); plt.ylabel("gCO2/kWh")
plt.tight_layout(); plt.show()''')

md("""## 4. Real workload — Google Cluster Trace 2011
Builds tasks from a downloaded chunk of the real Google `task_events` (submit + finish events →
arrival, duration, CPU request). The CPU request is already normalised 0–1 → used directly as utilisation.""")
code('''cols = ["time","missing","job_id","task_index","machine_id","event_type","user",
        "sched_class","priority","cpu_request","mem_request","disk_request","constraint"]
df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz", header=None, names=cols)
sub = (df[df.event_type==0][["job_id","task_index","time","cpu_request"]].dropna(subset=["cpu_request"])
       .rename(columns={"time":"submit"}).groupby(["job_id","task_index"], as_index=False).first())
end = (df[df.event_type.isin([2,3,4,5])][["job_id","task_index","time"]]
       .rename(columns={"time":"end"}).groupby(["job_id","task_index"], as_index=False).first())
m = sub.merge(end, on=["job_id","task_index"]); m["dur_us"] = m["end"] - m["submit"]
m = m[(m.dur_us>0) & (m.cpu_request>0)].reset_index(drop=True).head(60)
smin, smax = m.submit.min(), m.submit.max()
tasks = []
for _, r in m.iterrows():
    dur = int(np.clip(math.ceil(r.dur_us/1.8e9), 1, 12)); u = round(float(np.clip(r.cpu_request,0.05,1.0)),3)
    e = int((r.submit-smin)/(smax-smin+1)*(H//3)); tasks.append({"dur":dur,"u":u,"earliest":e,"deadline":e+dur+8})
print("%d real Google tasks. Sample:" % len(tasks)); pd.DataFrame(tasks).head(6)''')

md(r"""## 5. Metrics
$$ SLAV = \frac{\bigl|\{\,i: finish_i > deadline_i\,\}\bigr|}{N}\times 100\%. $$""")
code('''MAX_DEFER = 24
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
    return out''')

md(r"""## 6. Existing algorithms on a common benchmark
$$ \min_{s}\; F(s) = \frac{CO_2(s)}{CO_2^{\,base}} + 3\cdot\frac{SLAV(s)}{100}. $$""")
code('''base = evaluate([t["earliest"] for t in tasks])
def fitness(x):
    m = evaluate(decode(x))
    return (m["Carbon_kgCO2"]/base["Carbon_kgCO2"]) + 3.0*(m["SLA_viol_%"]/100.0)
ALGOS = {"GWO": GWO.OriginalGWO, "PSO": PSO.OriginalPSO, "DE": DE.OriginalDE,
         "WOA": WOA.OriginalWOA, "HHO": HHO.OriginalHHO, "GA": GA.OriginalGA}
results = {"FIFO/Round-Robin (baseline)": base}
for name, cls in ALGOS.items():
    problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*len(tasks), ub=[1.0]*len(tasks)),
               "minmax": "min", "log_to": None}
    g = cls(epoch=150, pop_size=50).solve(problem, seed=SEED)
    results[name] = evaluate(decode(g.solution)); print("simulated", name)''')

md("## 7. Comparison")
code('''df_res = pd.DataFrame(results).T.round(3)
df_res["CarbonReduction_%"] = ((base["Carbon_kgCO2"] - df_res["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100).round(1)
df_res.loc["FIFO/Round-Robin (baseline)", "CarbonReduction_%"] = 0.0
df_res''')
code('''algo = df_res.drop(index="FIFO/Round-Robin (baseline)")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(algo.index, algo["CarbonReduction_%"], color="tab:green"); ax[0].set(title="Carbon reduction vs baseline (%)", ylabel="%")
ax[1].bar(df_res.index, df_res["SLA_viol_%"], color="tab:orange"); ax[1].set(title="SLA violations (%)", ylabel="%")
ax[1].tick_params(axis="x", rotation=45); plt.tight_layout(); plt.show()''')

md("""## 8. Summary
All algorithms are simulated on the same real Google workload + carbon data (level playing field).
They cut carbon by shifting deferrable jobs into cleaner periods (energy unchanged). Next stages add
LSTM forecasting and battery/solar storage to the best scheduler.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "cas-venv", "display_name": "Python (cas)", "language": "python"}
nb["metadata"]["language_info"] = {"name": "python"}
out = ROOT + "/notebooks/carbon_aware_scheduling.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")

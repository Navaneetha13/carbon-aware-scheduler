"""Week 4 additions (on the real Google trace + real carbon):
  * multi-seed robustness  -> each metaheuristic run with 5 seeds; report mean +/- std
  * makespan metric        -> completion time of the schedule (hours)
  * carbon-aware baseline  -> greedy 'shift each job to its greenest feasible slot' (no metaheuristic)
All real data; no invented numbers.  (Energy-aware/consolidation baseline + utilisation % need a
host-capacity model and are handled separately.)"""
import json, math
import numpy as np
import pandas as pd
from mealpy import FloatVar, WOA, GWO, PSO, DE, HHO, GA

ROOT = "/home/durga/carbon-aware-scheduler"
SEEDS = [1, 2, 3, 4, 5]

# ---- energy model + real carbon + real Google tasks (same as the notebook) ----
P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5
def slot_energy_kwh(u): return (P_IDLE_W + (P_MAX_W - P_IDLE_W) * u) * SLOT_H / 1000.0
j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI)//48 + 1):
    for s in range(32, 40):
        k = day*48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)
cols = ["time","missing","job_id","task_index","machine_id","event_type","user","sched_class",
        "priority","cpu_request","mem_request","disk_request","constraint"]
df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz", header=None, names=cols)
sub = (df[df.event_type==0][["job_id","task_index","time","cpu_request"]].dropna(subset=["cpu_request"])
       .rename(columns={"time":"submit"}).groupby(["job_id","task_index"], as_index=False).first())
end = (df[df.event_type.isin([2,3,4,5])][["job_id","task_index","time"]]
       .rename(columns={"time":"end"}).groupby(["job_id","task_index"], as_index=False).first())
m = sub.merge(end, on=["job_id","task_index"]); m["dur_us"] = m["end"] - m["submit"]
m = m[(m.dur_us>0) & (m.cpu_request>0)].reset_index(drop=True).head(60)
smin, smax, SLACK, MAX_DEFER = m.submit.min(), m.submit.max(), 8, 24
tasks = []
for _, r in m.iterrows():
    dur = int(np.clip(math.ceil(r.dur_us/1.8e9),1,12)); u = float(np.clip(r.cpu_request,0.05,1.0))
    e = int((r.submit-smin)/(smax-smin+1)*(H//3)); tasks.append({"dur":dur,"u":u,"earliest":e,"deadline":e+dur+SLACK})
N = len(tasks)

# ---- metrics (now incl. makespan) ----
def evaluate(starts):
    energy = carbon_g = cost = 0.0; viol = 0; finish_max = 0
    for t, s in zip(tasks, starts):
        e = slot_energy_kwh(t["u"]); run = [k for k in range(s, s+t["dur"]) if k < H]
        energy += e*len(run); carbon_g += sum(e*CI[k] for k in run); cost += sum(e*PRICE[k] for k in run)
        finish_max = max(finish_max, s + t["dur"])
        if s + t["dur"] > t["deadline"]: viol += 1
    return {"Carbon_kgCO2": carbon_g/1000.0, "Energy_kWh": energy, "Cost_GBP": cost,
            "SLA_viol_%": 100.0*viol/N, "Makespan_h": finish_max * SLOT_H}

def decode(x):
    out = []
    for xi, t in zip(x, tasks):
        room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
        out.append(t["earliest"] + int(round(xi*room)))
    return out

base = evaluate([t["earliest"] for t in tasks])
def fitness(x):
    mm = evaluate(decode(x))
    return (mm["Carbon_kgCO2"]/base["Carbon_kgCO2"]) + 3.0*(mm["SLA_viol_%"]/100.0)
def cred(mm): return (base["Carbon_kgCO2"] - mm["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100.0

# ---- carbon-aware greedy baseline (each job -> greenest deadline-feasible slot) ----
def greedy_carbon_starts():
    starts = []
    for t in tasks:
        room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
        hi = max(0, min(room, t["deadline"] - t["dur"] - t["earliest"]))
        best_o, best_c = 0, float("inf")
        for o in range(0, hi+1):
            c = sum(CI[k] for k in range(t["earliest"]+o, t["earliest"]+o+t["dur"]) if k < H)
            if c < best_c: best_c, best_o = c, o
        starts.append(t["earliest"] + best_o)
    return starts

def carbon_aware_seeds(pop_size, r):
    g = np.array([(s - t["earliest"]) / max(1, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
                  for s, t in zip(greedy_carbon_starts(), tasks)])
    g = np.clip(g, 0, 1); seeds = [g.copy()]
    for _ in range(pop_size // 3):
        seeds.append(np.clip(g + r.normal(0, 0.10, N), 0, 1))
    while len(seeds) < pop_size:
        seeds.append(r.uniform(0, 1, N))
    return np.array(seeds[:pop_size])

# ---- multi-seed runs ----
def multiseed(cls, seeded=False):
    rows = []
    for sd in SEEDS:
        problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*N, ub=[1.0]*N), "minmax": "min", "log_to": None}
        starting = carbon_aware_seeds(50, np.random.default_rng(sd)) if seeded else None
        g = cls(epoch=150, pop_size=50).solve(problem, starting_solutions=starting, seed=sd)
        rows.append(evaluate(decode(g.solution)))
    return rows

print("Running multi-seed robustness (5 seeds each)...")
table = []
# deterministic baselines
table.append(("FIFO/Round-Robin (baseline)", 0.0, 0.0, base["SLA_viol_%"], 0.0, base["Makespan_h"]))
gca = evaluate(greedy_carbon_starts())
table.append(("Carbon-aware greedy (baseline)", cred(gca), 0.0, gca["SLA_viol_%"], 0.0, gca["Makespan_h"]))
# metaheuristics (5 seeds)
for name, cls, seeded in [("WOA", WOA.OriginalWOA, False), ("GWO", GWO.OriginalGWO, False),
                          ("PSO", PSO.OriginalPSO, False), ("DE", DE.OriginalDE, False),
                          ("HHO", HHO.OriginalHHO, False), ("GA", GA.OriginalGA, False),
                          ("CA-WOA", WOA.OriginalWOA, True)]:
    rows = multiseed(cls, seeded)
    creds = [cred(r) for r in rows]; slas = [r["SLA_viol_%"] for r in rows]; mk = [r["Makespan_h"] for r in rows]
    table.append((name, np.mean(creds), np.std(creds), np.mean(slas), np.std(slas), np.mean(mk)))
    print("done", name)

out = pd.DataFrame(table, columns=["Method","Carbon_red_mean_%","Carbon_red_std","SLA_mean_%","SLA_std","Makespan_h"]).round(2)
print("\n" + "="*92)
print("WEEK-4: multi-seed robustness (5 seeds) + makespan, on REAL Google trace + REAL carbon")
print("="*92)
print(out.to_string(index=False))
out.to_csv(ROOT + "/results/week4_robustness.csv", index=False)
print("\nSaved -> results/week4_robustness.csv")

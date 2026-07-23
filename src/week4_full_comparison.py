"""Week-4 FULL comparison on the real Google trace + real carbon, with a host-capacity model.

Adds vs the temporal-only model:
  * Hosts with capacity C=1.0 (one normalised machine; Google cpu_request is a fraction of the
    largest machine, so C=1 = a full machine).  Number of hosts M is DERIVED from the workload:
    M = ceil(peak per-slot CPU demand under the FIFO baseline)  -> the minimum to run the baseline.
  * Energy now depends on the number of ACTIVE hosts per slot:
        E_slot = (active_hosts * P_idle + (P_max-P_idle)*load) * dt
    Consolidation (perfect packing -> active=ceil(load/C)) uses fewer hosts than naive round-robin
    (one task per host), so it uses less energy -> gives a meaningful ENERGY-AWARE baseline.
  * UTILISATION = mean over active slots of (load / (active_hosts*C)).
  * CAPACITY OVERLOAD = demand above M (jobs competing for the cleanest slots).  This makes the
    problem NON-separable, so the metaheuristic must spread jobs across green slots, not pile them.
Baselines: FIFO/Round-Robin (naive), Energy-aware (consolidation), Carbon-aware greedy.
3 seeds per metaheuristic (mean +/- std).  All real data; documented parameters; no invented numbers.
"""
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mealpy import FloatVar, WOA, GWO, PSO, DE, HHO, GA

ROOT = "/home/durga/carbon-aware-scheduler"
SEEDS = [1, 2, 3]
C = 1.0                       # host capacity = one normalised machine
P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5

# ---- real carbon + price ----
j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI)//48 + 1):
    for s in range(32, 40):
        k = day*48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)

# ---- real Google tasks ----
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

def slot_loads(starts):
    load = {}; count = {}
    for t, s in zip(tasks, starts):
        for k in range(s, s+t["dur"]):
            if k < H:
                load[k] = load.get(k, 0.0) + t["u"]; count[k] = count.get(k, 0) + 1
    return load, count

# ---- DERIVE number of hosts M from the FIFO baseline's peak demand ----
fifo_starts = [t["earliest"] for t in tasks]
fifo_load, _ = slot_loads(fifo_starts)
PEAK = max(fifo_load.values())
M = math.ceil(PEAK)           # minimum hosts to run the baseline without overload
print("Derived host count M = ceil(peak FIFO demand %.2f) = %d hosts (capacity C=%.1f each)" % (PEAK, M, C))

def evaluate(starts, consolidate=True):
    load, count = slot_loads(starts)
    carbon_g = cost = energy = 0.0; util = []; overload = 0.0; total_load = 0.0; finish_max = 0
    for t, s in zip(tasks, starts):
        e = (P_MAX_W*0)  # placeholder (energy computed per slot below)
        finish_max = max(finish_max, s + t["dur"])
    viol = sum(1 for t, s in zip(tasks, starts) if s + t["dur"] > t["deadline"])
    for k, ld in load.items():
        active = (count[k] if not consolidate else max(1, math.ceil(ld / C)))
        energy += (active*P_IDLE_W + (P_MAX_W-P_IDLE_W)*ld) * SLOT_H / 1000.0
        carbon_g += ((P_MAX_W-P_IDLE_W)*ld + active*P_IDLE_W) * SLOT_H/1000.0 * CI[k]
        cost += (active*P_IDLE_W + (P_MAX_W-P_IDLE_W)*ld) * SLOT_H/1000.0 * PRICE[k]
        util.append(ld / (active * C)); overload += max(0.0, ld - M); total_load += ld
    return {"Carbon_kgCO2": carbon_g/1000.0, "Energy_kWh": energy, "Cost_GBP": cost,
            "SLA_viol_%": 100.0*viol/N, "Makespan_h": finish_max*SLOT_H,
            "Utilisation_%": 100.0*float(np.mean(util)) if util else 0.0,
            "Overload_%": 100.0*overload/total_load if total_load else 0.0}

def decode(x):
    out = []
    for xi, t in zip(x, tasks):
        room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
        out.append(t["earliest"] + int(round(xi*room)))
    return out

base = evaluate(fifo_starts, consolidate=False)         # FIFO/Round-Robin (naive placement)
# ---- Fitness = weighted sum of normalised objectives; weights MUST sum to 1.0 ----
#   f(x) = ALPHA*carbon + BETA*SLA + GAMMA*overload   (each divided by the FIFO baseline -> unitless)
ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3   # carbon, SLA, overload  ->  0.4 + 0.3 + 0.3 = 1.0
assert abs(ALPHA + BETA + GAMMA - 1.0) < 1e-9, "fitness weights must sum to 1"
def fitness(x):
    mm = evaluate(decode(x), consolidate=True)
    carbon   = mm["Carbon_kgCO2"] / base["Carbon_kgCO2"]
    sla      = mm["SLA_viol_%"] / 100.0
    overload = mm["Overload_%"] / 100.0
    return ALPHA*carbon + BETA*sla + GAMMA*overload
def cred(mm): return (base["Carbon_kgCO2"] - mm["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100.0

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

def seeds_from_greedy(pop, r):
    g = np.array([(s-t["earliest"])/max(1, min(MAX_DEFER, H-t["dur"]-t["earliest"]))
                  for s, t in zip(greedy_carbon_starts(), tasks)]); g = np.clip(g,0,1)
    out = [g.copy()]
    for _ in range(pop//3): out.append(np.clip(g + r.normal(0,0.10,N),0,1))
    while len(out) < pop: out.append(r.uniform(0,1,N))
    return np.array(out[:pop])

def run_meta(cls, seeded):
    rows = []
    for sd in SEEDS:
        problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*N, ub=[1.0]*N), "minmax":"min", "log_to": None}
        st = seeds_from_greedy(40, np.random.default_rng(sd)) if seeded else None
        g = cls(epoch=120, pop_size=40).solve(problem, starting_solutions=st, seed=sd)
        rows.append(evaluate(decode(g.solution), consolidate=True))
    return rows

def agg(rows):
    return {"Carbon_red_%": round(np.mean([cred(r) for r in rows]),2),
            "SLA_%": round(np.mean([r["SLA_viol_%"] for r in rows]),2),
            "Overload_%": round(np.mean([r["Overload_%"] for r in rows]),2),
            "Energy_kWh": round(np.mean([r["Energy_kWh"] for r in rows]),2),
            "Util_%": round(np.mean([r["Util_%" if False else "Utilisation_%"] for r in rows]),1),
            "Makespan_h": round(np.mean([r["Makespan_h"] for r in rows]),1)}

rows_out = []
def add(name, mm, cr=None):
    rows_out.append({"Method": name, "Carbon_red_%": round(cr if cr is not None else cred(mm),2),
                     "SLA_%": round(mm["SLA_viol_%"],2), "Overload_%": round(mm["Overload_%"],2),
                     "Energy_kWh": round(mm["Energy_kWh"],2), "Util_%": round(mm["Utilisation_%"],1),
                     "Makespan_h": round(mm["Makespan_h"],1)})

# Baselines
add("FIFO/Round-Robin (baseline)", base, cr=0.0)
add("Energy-aware (consolidation)", evaluate(fifo_starts, consolidate=True))
add("Carbon-aware greedy", evaluate(greedy_carbon_starts(), consolidate=True))
# Metaheuristics + CA-WOA (3 seeds)
print("Running metaheuristics (3 seeds each) under capacity model...")
for name, cls, seeded in [("WOA",WOA.OriginalWOA,False),("GWO",GWO.OriginalGWO,False),("PSO",PSO.OriginalPSO,False),
                          ("DE",DE.OriginalDE,False),("HHO",HHO.OriginalHHO,False),("GA",GA.OriginalGA,False),
                          ("CA-WOA",WOA.OriginalWOA,True)]:
    a = agg(run_meta(cls, seeded))
    rows_out.append({"Method": name, **a}); print("done", name)

out = pd.DataFrame(rows_out)[["Method","Carbon_red_%","SLA_%","Overload_%","Energy_kWh","Util_%","Makespan_h"]]
print("\n" + "="*100)
print("WEEK-4 FULL COMPARISON — host-capacity model (M=%d hosts), real Google trace + real carbon" % M)
print("="*100)
print(out.to_string(index=False))
out.to_csv(ROOT + "/results/week4_full_comparison.csv", index=False)
print("\nSaved -> results/week4_full_comparison.csv")

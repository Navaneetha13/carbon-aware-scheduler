"""Scalability of the host-capacity model across workload sizes.

Evaluates every method at N = 50, 100, 200 and 300 tasks with five independent
seeds per point, so the trend can be reported with mean and standard deviation.

Mirrors week4_full_comparison.py (same energy model, same normalised fitness with
alpha/beta/gamma = 0.4/0.3/0.3, same runtime-derived host count, same Mealpy
defaults) and is parameterised by task count.

Writes results/scalability_sweep.csv
"""
import json, math, os, sys
import numpy as np
import pandas as pd
from mealpy import FloatVar, WOA, GWO, PSO, DE, HHO, GA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [1, 2, 3, 4, 5]
TASK_COUNTS = [50, 100, 200, 300]
C = 1.0
P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5

# ---- real carbon + price (identical to week4_full_comparison.py) ----
j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI) // 48 + 1):
    for s in range(32, 40):
        k = day * 48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)

# ---- real Google tasks (identical parsing) ----
cols = ["time","missing","job_id","task_index","machine_id","event_type","user","sched_class",
        "priority","cpu_request","mem_request","disk_request","constraint"]
df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz", header=None, names=cols)
sub = (df[df.event_type == 0][["job_id","task_index","time","cpu_request"]].dropna(subset=["cpu_request"])
       .rename(columns={"time":"submit"}).groupby(["job_id","task_index"], as_index=False).first())
end = (df[df.event_type.isin([2,3,4,5])][["job_id","task_index","time"]]
       .rename(columns={"time":"end"}).groupby(["job_id","task_index"], as_index=False).first())
merged = sub.merge(end, on=["job_id","task_index"])
merged["dur_us"] = merged["end"] - merged["submit"]
merged = merged[(merged.dur_us > 0) & (merged.cpu_request > 0)].reset_index(drop=True)
SLACK, MAX_DEFER = 8, 24


def build_tasks(n):
    m = merged.head(n)
    smin, smax = m.submit.min(), m.submit.max()
    out = []
    for _, r in m.iterrows():
        dur = int(np.clip(math.ceil(r.dur_us / 1.8e9), 1, 12))
        u = float(np.clip(r.cpu_request, 0.05, 1.0))
        e = int((r.submit - smin) / (smax - smin + 1) * (H // 3))
        out.append({"dur": dur, "u": u, "earliest": e, "deadline": e + dur + SLACK})
    return out


def make_env(tasks):
    N = len(tasks)

    def slot_loads(starts):
        load, count = {}, {}
        for t, s in zip(tasks, starts):
            for k in range(s, s + t["dur"]):
                if k < H:
                    load[k] = load.get(k, 0.0) + t["u"]
                    count[k] = count.get(k, 0) + 1
        return load, count

    fifo_starts = [t["earliest"] for t in tasks]
    fifo_load, _ = slot_loads(fifo_starts)
    M = math.ceil(max(fifo_load.values()))

    def evaluate(starts, consolidate=True):
        load, count = slot_loads(starts)
        carbon_g = cost = energy = 0.0
        util, overload, total_load, finish_max = [], 0.0, 0.0, 0
        for t, s in zip(tasks, starts):
            finish_max = max(finish_max, s + t["dur"])
        viol = sum(1 for t, s in zip(tasks, starts) if s + t["dur"] > t["deadline"])
        for k, ld in load.items():
            active = (count[k] if not consolidate else max(1, math.ceil(ld / C)))
            energy += (active * P_IDLE_W + (P_MAX_W - P_IDLE_W) * ld) * SLOT_H / 1000.0
            carbon_g += ((P_MAX_W - P_IDLE_W) * ld + active * P_IDLE_W) * SLOT_H / 1000.0 * CI[k]
            cost += (active * P_IDLE_W + (P_MAX_W - P_IDLE_W) * ld) * SLOT_H / 1000.0 * PRICE[k]
            util.append(ld / (active * C))
            overload += max(0.0, ld - M)
            total_load += ld
        return {"Carbon_kgCO2": carbon_g / 1000.0, "Energy_kWh": energy, "Cost_GBP": cost,
                "SLA_viol_%": 100.0 * viol / N, "Makespan_h": finish_max * SLOT_H,
                "Utilisation_%": 100.0 * float(np.mean(util)) if util else 0.0,
                "Overload_%": 100.0 * overload / total_load if total_load else 0.0}

    def decode(x):
        out = []
        for xi, t in zip(x, tasks):
            room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
            out.append(t["earliest"] + int(round(xi * room)))
        return out

    base = evaluate(fifo_starts, consolidate=False)
    ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
    assert abs(ALPHA + BETA + GAMMA - 1.0) < 1e-9

    def fitness(x):
        mm = evaluate(decode(x), consolidate=True)
        return (ALPHA * mm["Carbon_kgCO2"] / base["Carbon_kgCO2"]
                + BETA * mm["SLA_viol_%"] / 100.0
                + GAMMA * mm["Overload_%"] / 100.0)

    def cred(mm):
        return (base["Carbon_kgCO2"] - mm["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100.0

    def greedy_carbon_starts():
        starts = []
        for t in tasks:
            room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
            hi = max(0, min(room, t["deadline"] - t["dur"] - t["earliest"]))
            best_o, best_c = 0, float("inf")
            for o in range(0, hi + 1):
                c = sum(CI[k] for k in range(t["earliest"] + o, t["earliest"] + o + t["dur"]) if k < H)
                if c < best_c: best_c, best_o = c, o
            starts.append(t["earliest"] + best_o)
        return starts

    def seeds_from_greedy(pop, rng):
        g = np.array([(s - t["earliest"]) / max(1, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
                      for s, t in zip(greedy_carbon_starts(), tasks)])
        g = np.clip(g, 0, 1)
        out = [g.copy()]
        for _ in range(pop // 3):
            out.append(np.clip(g + rng.normal(0, 0.10, N), 0, 1))
        while len(out) < pop:
            out.append(rng.uniform(0, 1, N))
        return np.array(out[:pop])

    return N, M, evaluate, decode, fitness, cred, seeds_from_greedy


rows = []
for n in TASK_COUNTS:
    tasks = build_tasks(n)
    N, M, evaluate, decode, fitness, cred, seeds_from_greedy = make_env(tasks)
    print(f"\n=== N={N} tasks | derived hosts M={M} ===", flush=True)
    for name, cls, seeded in [("WOA", WOA.OriginalWOA, False), ("GWO", GWO.OriginalGWO, False),
                              ("PSO", PSO.OriginalPSO, False), ("DE", DE.OriginalDE, False),
                              ("HHO", HHO.OriginalHHO, False), ("GA", GA.OriginalGA, False),
                              ("CA-WOA", WOA.OriginalWOA, True)]:
        crs, slas = [], []
        for sd in SEEDS:
            problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*N, ub=[1.0]*N),
                       "minmax": "min", "log_to": None}
            st = seeds_from_greedy(40, np.random.default_rng(sd)) if seeded else None
            g = cls(epoch=120, pop_size=40).solve(problem, starting_solutions=st, seed=sd)
            mm = evaluate(decode(g.solution), consolidate=True)
            crs.append(cred(mm)); slas.append(mm["SLA_viol_%"])
        rows.append({"N": N, "M": M, "Method": name,
                     "Carbon_red_%": round(float(np.mean(crs)), 2),
                     "Carbon_red_std": round(float(np.std(crs)), 2),
                     "SLA_%": round(float(np.mean(slas)), 2),
                     "SLA_std": round(float(np.std(slas)), 2)})
        print(f"  {name:8} carbon_red {np.mean(crs):6.2f}% (sd {np.std(crs):.2f})   "
              f"SLA {np.mean(slas):6.2f}% (sd {np.std(slas):.2f})", flush=True)

out = pd.DataFrame(rows)
out.to_csv(ROOT + "/results/scalability_sweep.csv", index=False)
print("\nSaved -> results/scalability_sweep.csv")
print(out.pivot(index="Method", columns="N", values="Carbon_red_%").to_string())
print()
print(out.pivot(index="Method", columns="N", values="SLA_%").to_string())

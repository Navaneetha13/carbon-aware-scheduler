"""PROPOSED METHOD — CA-WOA: a Carbon-Aware (enhanced) Whale Optimization Algorithm.

Novelty: WOA has only been used for energy/cost scheduling in the literature, never for CARBON.
Here we apply WOA to carbon-aware scheduling AND enhance it with a domain-specific improvement:
  * Carbon-aware seeding  -> part of WOA's initial population is built by shifting each job toward
    its lowest-carbon, deadline-feasible slot (instead of pure random), giving the search a head start.

Everything runs on REAL data (Google Cluster Trace + UK National Grid carbon). No invented results.
"""
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mealpy import FloatVar, WOA, GWO, PSO, DE, HHO, GA

ROOT = "/home/durga/carbon-aware-scheduler"; SEED = 42
rng = np.random.default_rng(SEED)

# ---------- energy model ----------
P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5
def slot_energy_kwh(u): return (P_IDLE_W + (P_MAX_W - P_IDLE_W) * u) * SLOT_H / 1000.0

# ---------- real carbon + price ----------
j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI)//48 + 1):
    for s in range(32, 40):
        k = day*48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)

# ---------- real Google workload ----------
cols = ["time","missing","job_id","task_index","machine_id","event_type","user","sched_class",
        "priority","cpu_request","mem_request","disk_request","constraint"]
df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz", header=None, names=cols)
sub = (df[df.event_type == 0][["job_id","task_index","time","cpu_request"]].dropna(subset=["cpu_request"])
       .rename(columns={"time":"submit"}).groupby(["job_id","task_index"], as_index=False).first())
end = (df[df.event_type.isin([2,3,4,5])][["job_id","task_index","time"]]
       .rename(columns={"time":"end"}).groupby(["job_id","task_index"], as_index=False).first())
m = sub.merge(end, on=["job_id","task_index"]); m["dur_us"] = m["end"] - m["submit"]
m = m[(m.dur_us > 0) & (m.cpu_request > 0)].reset_index(drop=True).head(60)
smin, smax, SLACK, MAX_DEFER = m.submit.min(), m.submit.max(), 8, 24
tasks = []
for _, r in m.iterrows():
    dur = int(np.clip(math.ceil(r.dur_us/1.8e9), 1, 12)); u = float(np.clip(r.cpu_request, 0.05, 1.0))
    e = int((r.submit - smin)/(smax - smin + 1) * (H//3))
    tasks.append({"dur": dur, "u": u, "earliest": e, "deadline": e + dur + SLACK})
N = len(tasks)

# ---------- metrics + schedulers ----------
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
    mm = evaluate(decode(x))
    return (mm["Carbon_kgCO2"]/base["Carbon_kgCO2"]) + 3.0*(mm["SLA_viol_%"]/100.0)

# ---------- CA-WOA enhancement: carbon-aware seeding ----------
def greedy_carbon_x():
    """For each task, choose the deadline-feasible start slot with the lowest carbon, encode to [0,1]."""
    x = np.zeros(N)
    for i, t in enumerate(tasks):
        room = max(0, min(MAX_DEFER, H - t["dur"] - t["earliest"]))
        if room == 0:
            x[i] = 0.0; continue
        hi = max(0, min(room, t["deadline"] - t["dur"] - t["earliest"]))   # keep SLA (finish<=deadline)
        best_o, best_c = 0, float("inf")
        for o in range(0, hi + 1):
            s = t["earliest"] + o
            c = sum(CI[k] for k in range(s, s + t["dur"]) if k < H)
            if c < best_c:
                best_c, best_o = c, o
        x[i] = best_o / room
    return np.clip(x, 0.0, 1.0)

def carbon_aware_seeds(pop_size):
    g = greedy_carbon_x()
    seeds = [g.copy()]                                  # 1 carbon-greedy seed
    for _ in range(pop_size // 3):                      # ~1/3 perturbed greedy (diversity near good region)
        seeds.append(np.clip(g + rng.normal(0, 0.10, N), 0.0, 1.0))
    while len(seeds) < pop_size:                        # rest random (exploration)
        seeds.append(rng.uniform(0.0, 1.0, N))
    return np.array(seeds[:pop_size])

# ---------- run ----------
def run(model, label, starting=None):
    problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*N, ub=[1.0]*N), "minmax": "min", "log_to": None}
    g = model.solve(problem, starting_solutions=starting, seed=SEED)
    r = evaluate(decode(g.solution)); print("ran", label)
    return r

EPOCH, POP = 150, 50
results = {"FIFO/Round-Robin (baseline)": base}
# PROPOSED: CA-WOA = WOA + carbon-aware seeding
results["CA-WOA (PROPOSED)"] = run(WOA.OriginalWOA(epoch=EPOCH, pop_size=POP), "CA-WOA (proposed)",
                                   starting=carbon_aware_seeds(POP))
# standard algorithms (random init) for comparison
for label, model in [("WOA (standard)", WOA.OriginalWOA(epoch=EPOCH, pop_size=POP)),
                     ("GWO", GWO.OriginalGWO(epoch=EPOCH, pop_size=POP)),
                     ("PSO", PSO.OriginalPSO(epoch=EPOCH, pop_size=POP)),
                     ("DE",  DE.OriginalDE(epoch=EPOCH, pop_size=POP)),
                     ("HHO", HHO.OriginalHHO(epoch=EPOCH, pop_size=POP)),
                     ("GA",  GA.OriginalGA(epoch=EPOCH, pop_size=POP))]:
    results[label] = run(model, label)

out = pd.DataFrame(results).T.round(3)
out["CarbonReduction_%"] = ((base["Carbon_kgCO2"] - out["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100).round(1)
out.loc["FIFO/Round-Robin (baseline)", "CarbonReduction_%"] = 0.0
print("\n" + "="*82)
print("PROPOSED CA-WOA vs standard algorithms — REAL Google trace + REAL carbon (level field)")
print("="*82)
print(out.to_string())
out.to_csv(ROOT + "/results/proposed_ca_woa_comparison.csv")

order = [i for i in out.index if i != "FIFO/Round-Robin (baseline)"]
colors = ["tab:green" if i == "CA-WOA (PROPOSED)" else "tab:blue" for i in order]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].bar(order, out.loc[order, "CarbonReduction_%"], color=colors)
ax[0].set(title="Carbon reduction vs baseline (%) — PROPOSED in green", ylabel="%"); ax[0].tick_params(axis="x", rotation=30)
ax[1].bar(out.index, out["SLA_viol_%"], color="tab:orange")
ax[1].set(title="SLA violations (%)", ylabel="%"); ax[1].tick_params(axis="x", rotation=40)
fig.tight_layout(); fig.savefig(ROOT + "/results/proposed_ca_woa_comparison.png", dpi=120)
print("\nSaved -> results/proposed_ca_woa_comparison.csv and .png")

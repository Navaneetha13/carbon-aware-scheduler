"""Simulating existing algorithms on the REAL Google Cluster Trace 2011 (a downloaded chunk).
Builds tasks from task_events (submit + finish events -> arrival, duration, cpu/mem request),
then runs GWO/PSO/DE/WOA/HHO/GA on the same carbon-aware scheduling problem + real carbon data."""
import gzip, math
import numpy as np
import pandas as pd
from mealpy import FloatVar, GWO, PSO, DE, WOA, HHO, GA

ROOT = "/home/durga/carbon-aware-scheduler"; SEED = 42

# ---- energy + carbon + price (same model as before) ----
P_IDLE_W, P_MAX_W, SLOT_H = 100.0, 250.0, 0.5
def slot_energy_kwh(u): return (P_IDLE_W + (P_MAX_W - P_IDLE_W) * u) * SLOT_H / 1000.0
import json
j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast") for r in j["data"]], float)
PRICE = np.full(len(CI), 0.15)
for day in range(len(CI)//48 + 1):
    for s in range(32, 40):
        k = day*48 + s
        if k < len(PRICE): PRICE[k] = 0.30
H = len(CI)

# ---- REAL Google Cluster Trace tasks ----
cols = ["time","missing","job_id","task_index","machine_id","event_type","user",
        "sched_class","priority","cpu_request","mem_request","disk_request","constraint"]
df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz", header=None, names=cols)
sub = (df[df.event_type == 0][["job_id","task_index","time","cpu_request"]]
       .dropna(subset=["cpu_request"]).rename(columns={"time":"submit"})
       .groupby(["job_id","task_index"], as_index=False).first())
end = (df[df.event_type.isin([2,3,4,5])][["job_id","task_index","time"]]
       .rename(columns={"time":"end"}).groupby(["job_id","task_index"], as_index=False).first())
m = sub.merge(end, on=["job_id","task_index"])
m["dur_us"] = m["end"] - m["submit"]
m = m[(m.dur_us > 0) & (m.cpu_request > 0)].reset_index(drop=True).head(60)

smin, smax, SLACK = m.submit.min(), m.submit.max(), 8
US_PER_SLOT = 1.8e9                                   # 30 min in microseconds
tasks = []
for _, r in m.iterrows():
    dur = int(np.clip(math.ceil(r.dur_us / US_PER_SLOT), 1, 12))
    u   = float(np.clip(r.cpu_request, 0.05, 1.0))    # Google cpu_request already normalised 0-1
    e   = int((r.submit - smin) / (smax - smin + 1) * (H // 3))
    tasks.append({"dur": dur, "u": round(u, 3), "earliest": e, "deadline": e + dur + SLACK})
print("REAL Google Cluster Trace -> %d tasks built (cpu u %.2f-%.2f, dur %d-%d slots)" %
      (len(tasks), min(t["u"] for t in tasks), max(t["u"] for t in tasks),
       min(t["dur"] for t in tasks), max(t["dur"] for t in tasks)))

# ---- metrics / schedulers (same as the NASA run) ----
MAX_DEFER = 24
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
base = evaluate([t["earliest"] for t in tasks])
def fitness(x):
    mm = evaluate(decode(x))
    return (mm["Carbon_kgCO2"]/base["Carbon_kgCO2"]) + 3.0*(mm["SLA_viol_%"]/100.0)

ALGOS = {"GWO": GWO.OriginalGWO, "PSO": PSO.OriginalPSO, "DE": DE.OriginalDE,
         "WOA": WOA.OriginalWOA, "HHO": HHO.OriginalHHO, "GA": GA.OriginalGA}
results = {"FIFO/Round-Robin (baseline)": base}
for name, cls in ALGOS.items():
    problem = {"obj_func": fitness, "bounds": FloatVar(lb=[0.0]*len(tasks), ub=[1.0]*len(tasks)),
               "minmax": "min", "log_to": None}
    g = cls(epoch=150, pop_size=50).solve(problem, seed=SEED)
    results[name] = evaluate(decode(g.solution))

out = pd.DataFrame(results).T.round(3)
out["CarbonReduction_%"] = ((base["Carbon_kgCO2"] - out["Carbon_kgCO2"]) / base["Carbon_kgCO2"] * 100).round(1)
out.loc["FIFO/Round-Robin (baseline)", "CarbonReduction_%"] = 0.0
print("\n=== SIMULATING EXISTING ALGORITHMS on REAL GOOGLE CLUSTER TRACE ===")
print(out.to_string())

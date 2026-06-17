"""
Week 3 demo — Carbon-aware temporal scheduling of a REAL workload with a metaheuristic.
MSc Cloud Computing, NCI — Navaneetha Thalakokkula.

REAL DATA (no synthetic/demo data):
  * Workload : NASA-iPSC-1993 trace, Parallel Workloads Archive (SWF format).
  * Carbon   : UK National Grid ESO Carbon Intensity API (gCO2/kWh, 30-min slots).
PLATFORM: Python 3.10 + Mealpy 3.0.3 (Grey Wolf Optimizer).

IDEA: the SAME jobs run either way, so total ENERGY is unchanged by *when* they run.
What temporal shifting changes is CARBON (different grid intensity per slot), COST
(time-of-use price) and SLA (deferring too far misses deadlines). GWO shifts deferrable
jobs into low-carbon slots while respecting deadlines; FIFO runs them at the earliest slot.
"""
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mealpy import FloatVar, GWO

ROOT = "/home/durga/carbon-aware-scheduler"
SEED = 42

# ===================== 1. ENERGY / POWER MODEL (SPECpower-style) =====================
P_IDLE_W = 100.0          # idle server power (W) — typical SPECpower figure (assumption)
P_MAX_W  = 250.0          # peak server power (W)
SLOT_H   = 0.5            # each scheduling slot = half an hour (matches carbon data)

def host_power_w(u):                       # (2)  P(u) = P_idle + (P_max - P_idle)*u
    return P_IDLE_W + (P_MAX_W - P_IDLE_W) * u

def slot_energy_kwh(u):                    # (3)  energy in ONE half-hour slot at utilisation u
    return host_power_w(u) * SLOT_H / 1000.0

# ===================== 2. REAL CARBON INTENSITY + price =====================
def load_carbon():
    j = json.load(open(f"{ROOT}/data/carbon/3day_window.json"))
    ci = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast")
                   for r in j["data"]], dtype=float)          # gCO2/kWh, 144 real slots
    # Time-of-use electricity price (£/kWh) — a documented tariff ASSUMPTION, not data:
    # peak 16:00-20:00 each day = 0.30, otherwise 0.15.
    price = np.full(len(ci), 0.15)
    for day in range(len(ci) // 48 + 1):
        for s in range(32, 40):                                # slots 16:00-20:00
            k = day * 48 + s
            if k < len(price):
                price[k] = 0.30
    return ci, price

# ===================== 3. REAL WORKLOAD (NASA-iPSC SWF) =====================
def load_tasks(n_tasks, horizon):
    cols = ["job","submit","wait","runtime","nproc","avg_cpu","used_mem","req_proc",
            "req_time","req_mem","status","uid","gid","app","queue","partition","prev_job","think"]
    df = pd.read_csv(f"{ROOT}/data/workload/NASA.swf", sep=r"\s+", comment=";",
                     header=None, names=cols)
    df = df[(df.runtime > 0) & (df.nproc > 0)].head(n_tasks).reset_index(drop=True)
    max_nproc = df.nproc.max()
    # compress real submit times into the first third of the horizon so jobs can be deferred
    smin, smax = df.submit.min(), df.submit.max()
    arrival_window = horizon // 3
    tasks = []
    for _, r in df.iterrows():
        dur = int(np.clip(math.ceil(r.runtime / 1800.0), 1, 12))      # runtime -> #slots (<=6h)
        u   = float(np.clip(r.nproc / max_nproc, 0.05, 1.0))          # CPU demand -> utilisation
        earliest = int((r.submit - smin) / (smax - smin + 1) * arrival_window)
        slack = 8                                                     # allowed deadline slack (4h)
        tasks.append({"dur": dur, "u": u, "earliest": earliest,
                      "deadline": earliest + dur + slack})
    return tasks

MAX_DEFER = 24            # a job may be deferred up to 24 slots (12h) to chase clean energy

# ===================== 4. METRICS =====================
def evaluate(starts, tasks, ci, price):
    energy = carbon_g = cost = 0.0
    violations = 0
    H = len(ci)
    for t, s in zip(tasks, starts):
        e_slot = slot_energy_kwh(t["u"])
        run = [k for k in range(s, s + t["dur"]) if k < H]
        energy   += e_slot * len(run)                       # (3) total energy
        carbon_g += sum(e_slot * ci[k]    for k in run)     # (4) energy * carbon intensity
        cost     += sum(e_slot * price[k] for k in run)     # (5) energy * price
        if s + t["dur"] > t["deadline"]:                    # (7) finished after deadline
            violations += 1
    return {"energy_kwh": energy, "carbon_kg": carbon_g / 1000.0, "cost_gbp": cost,
            "sla_viol_pct": 100.0 * violations / len(tasks)}

# ===================== 5. SCHEDULES =====================
def fifo_starts(tasks):                                     # baseline: earliest possible slot
    return [t["earliest"] for t in tasks]

def decode(x, tasks, H):                                    # x in [0,1]^N  ->  start slot per task
    starts = []
    for xi, t in zip(x, tasks):
        room = min(MAX_DEFER, H - t["dur"] - t["earliest"])
        room = max(0, room)
        starts.append(t["earliest"] + int(round(xi * room)))
    return starts

def make_fitness(tasks, ci, price, base_carbon, w_carbon=1.0, lam=0.5):
    # (10) minimise normalised carbon + penalty * SLA-violation fraction
    H = len(ci)
    def fitness(x):
        m = evaluate(decode(x, tasks, H), tasks, ci, price)
        return w_carbon * (m["carbon_kg"] / base_carbon) + lam * (m["sla_viol_pct"] / 100.0)
    return fitness

# ===================== RUN =====================
def main():
    ci, price = load_carbon()
    H = len(ci)
    tasks = load_tasks(n_tasks=120, horizon=H)
    print(f"Loaded {len(tasks)} real NASA tasks | carbon horizon = {H} slots "
          f"(CI {ci.min():.0f}-{ci.max():.0f} gCO2/kWh)\n")

    # Baseline: FIFO
    fifo = evaluate(fifo_starts(tasks), tasks, ci, price)

    # Proposed: GWO carbon-aware temporal shifting
    fit = make_fitness(tasks, ci, price, base_carbon=fifo["carbon_kg"])
    problem = {"obj_func": fit, "bounds": FloatVar(lb=[0.0]*len(tasks), ub=[1.0]*len(tasks)),
               "minmax": "min", "log_to": None}
    model = GWO.OriginalGWO(epoch=60, pop_size=40)
    g = model.solve(problem, seed=SEED)
    gwo = evaluate(decode(g.solution, tasks, H), tasks, ci, price)

    # Results
    def pct(a, b): return (a - b) / a * 100.0 if a else 0.0
    print("Metric            FIFO baseline     GWO (carbon-aware)    Change")
    print("-" * 66)
    print(f"Energy (kWh)      {fifo['energy_kwh']:>10.3f}        {gwo['energy_kwh']:>10.3f}"
          f"        {pct(fifo['energy_kwh'],gwo['energy_kwh']):>+6.1f}%")
    print(f"Carbon (kg CO2)   {fifo['carbon_kg']:>10.3f}        {gwo['carbon_kg']:>10.3f}"
          f"        {pct(fifo['carbon_kg'],gwo['carbon_kg']):>+6.1f}%")
    print(f"Cost (GBP)        {fifo['cost_gbp']:>10.3f}        {gwo['cost_gbp']:>10.3f}"
          f"        {pct(fifo['cost_gbp'],gwo['cost_gbp']):>+6.1f}%")
    print(f"SLA violations(%) {fifo['sla_viol_pct']:>10.2f}        {gwo['sla_viol_pct']:>10.2f}")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(range(H), ci, color="tab:red")
    ax[0].set(title="Real UK grid carbon intensity (3 days)", xlabel="half-hour slot",
              ylabel="gCO2/kWh")
    labels = ["Carbon\n(kg CO2)", "Cost\n(GBP)"]
    ax[1].bar([0,1], [fifo["carbon_kg"], fifo["cost_gbp"]], width=0.35, label="FIFO")
    ax[1].bar([0.4,1.4], [gwo["carbon_kg"], gwo["cost_gbp"]], width=0.35, label="GWO carbon-aware")
    ax[1].set_xticks([0.2,1.2]); ax[1].set_xticklabels(labels)
    ax[1].set(title="FIFO vs GWO"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{ROOT}/results/week3_demo.png", dpi=120)
    print(f"\nSaved figure -> results/week3_demo.png")

if __name__ == "__main__":
    main()

"""Experiment runner. Every row is one real run; nothing is hardcoded.

Usage:  python runner.py E1 E4 E10 ...
Writes: raw_<EXP>.csv  (one row per config x method x seed), resumable.
"""
import itertools, os, sys, time, tracemalloc
import numpy as np
import pandas as pd
from multiprocessing import Pool

import core as K
import baselines as B
from mealpy import FloatVar, WOA, GWO, PSO, DE, HHO, GA

OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(1, 31))          # 30 independent seeds
POP, EPOCH = 40, 120
NS_MAIN = [500, 1000, 1500, 2000, 2500, 3000]
MS = [10, 20]

# GA.OriginalGA performs no search in mealpy 3.0.3 (40 objective evaluations
# instead of 4840); GA.BaseGA is the working implementation. The defect is
# measured explicitly in the GABUG experiment.
ALGOS = {"WOA": WOA.OriginalWOA, "GWO": GWO.OriginalGWO, "PSO": PSO.OriginalPSO,
         "DE": DE.OriginalDE, "HHO": HHO.OriginalHHO, "GA": GA.BaseGA}

_CACHE = {}
KEYS = ("wl", "N", "M", "power", "hard", "cap", "pen", "alpha", "beta", "gamma",
        "subset", "mmin", "region", "window", "fleet", "startup")


def get_env(job):
    key = tuple(job[k] for k in KEYS)
    if key not in _CACHE:
        _CACHE[key] = K.make_env(workload=job["wl"], n=job["N"], M=job["M"],
                                 power=job["power"], hard=job["hard"], cap=job["cap"],
                                 subset=job["subset"], alpha=job["alpha"],
                                 beta=job["beta"], gamma=job["gamma"], pen=job["pen"],
                                 mmin=job["mmin"], region=job["region"],
                                 window=job["window"], fleet=job["fleet"],
                                 startup=job["startup"])
    return _CACHE[key]


def one(job):
    env = get_env(job)
    row = dict(job)
    row.update({"peak_demand": round(env.peak, 2),
                "total_load": round(float((env.u * env.dur).sum()), 2),
                "arrival_slots": int(np.unique(env.earliest).size)})
    method, init = job["method"], job["init"]

    if method == "FIFO":
        mm, nfe, el, mem = env.base, 0, 0.0, 0.0
    elif method == "Consolidation":
        mm, nfe, el, mem = env.cons, 0, 0.0, 0.0
    elif method in B.POLICIES:
        t0 = time.time()
        mm = env.evaluate(np.asarray(B.POLICIES[method](env), int), consolidate=True)
        el, nfe, mem = time.time() - t0, 0, 0.0
    elif method == "Greedy(EDF+greenest)":
        t0 = time.time()
        mm = env.evaluate(env.greedy_starts(), consolidate=True)
        el, nfe, mem = time.time() - t0, 0, 0.0
    else:
        cls = GA.OriginalGA if method == "GA_OriginalGA" else ALGOS[method]
        rng = np.random.default_rng(job["seed"])
        start = (env.pop_improved(POP, rng) if init == "improved" else
                 env.pop_carbon(POP, rng, frac=job["seed_frac"]) if init == "carbon" else None)
        env.nfe = 0
        problem = {"obj_func": env.fitness,
                   "bounds": FloatVar(lb=[0.0] * env.N, ub=[1.0] * env.N),
                   "minmax": "min", "log_to": None}
        tracemalloc.start()
        t0 = time.time()
        model = cls(epoch=EPOCH, pop_size=POP)
        res = model.solve(problem, starting_solutions=start, seed=job["seed"])
        el = time.time() - t0
        if job.get("curve"):
            row["curve"] = ";".join("%.6f" % v for v in
                                    model.history.list_global_best_fit)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem, nfe = peak / 1048576.0, env.nfe
        mm = env.evaluate(env.decode(res.solution), consolidate=True)

    row.update({"carbon_kg": round(mm["Carbon_kgCO2"], 6),
                "carbon_red_vs_naive_%": round(env.cred(mm), 4),
                "carbon_red_vs_consol_%": round(env.cred_vs_cons(mm), 4),
                "sla_%": round(mm["SLA_%"], 4), "viol_n": mm.get("Viol_n", 0),
                "energy_kwh": round(mm["Energy_kWh"], 4),
                "cost_gbp": round(mm["Cost_GBP"], 4),
                "util_%": round(mm["Util_%"], 3),
                "overload_%": round(mm["Overload_%"], 4),
                "feasible": bool(mm.get("Feasible", True)),
                "makespan_h": round(mm["Makespan_h"], 2),
                "nfe": nfe, "runtime_s": round(el, 4), "peak_mem_mb": round(mem, 2)})
    return row


def J(**kw):
    d = dict(wl="google", N=1000, M=10, power="linear", hard=False, cap=False, pen=0.0,
             alpha=0.4, beta=0.3, gamma=0.3, subset=0, mmin=0,
             region="UK", window=None, fleet="uniform", startup="none",
             method="WOA", init="random", seed=1, seed_frac=1 / 3, exp="")
    d.update(kw)
    return d


DET = ("FIFO", "Consolidation", "Greedy(EDF+greenest)")


def grid(exp, cap=False, hard=False, ns=None, pen=0.0):
    """Standard sweep: deterministic baselines + 6 optimisers + CA-WOA."""
    out = []
    for n, M in itertools.product(ns or NS_MAIN, MS):
        for m in DET:
            out.append(J(exp=exp, N=n, M=M, cap=cap, hard=hard, pen=pen, method=m,
                         seed=0, init="-"))
        for a in ALGOS:
            for s in SEEDS:
                out.append(J(exp=exp, N=n, M=M, cap=cap, hard=hard, pen=pen, method=a,
                             init="random", seed=s))
        for s in SEEDS:                      # CA-WOA = WOA + carbon-aware seeding
            out.append(J(exp=exp, N=n, M=M, cap=cap, hard=hard, pen=pen, method="WOA",
                         init="carbon", seed=s))
    return out


def ablation(exp, cap, ns, pen=0.0):
    out = []
    for n, M in itertools.product(ns, MS):
        hard = pen > 0
        out.append(J(exp=exp, N=n, M=M, cap=cap, pen=pen, hard=hard,
                     method="Greedy(EDF+greenest)", seed=0, init="-"))
        for init in ("random", "improved", "carbon"):
            for s in SEEDS:
                out.append(J(exp=exp, N=n, M=M, cap=cap, pen=pen, hard=hard,
                             method="WOA", init=init, seed=s))
        for (b, g) in ((0.0, 0.3), (0.3, 0.0)):     # drop SLA term / drop penalty term
            for s in SEEDS:
                out.append(J(exp=exp, N=n, M=M, cap=cap, pen=pen, hard=hard,
                             method="WOA", init="carbon",
                             beta=b, gamma=g, alpha=1.0 - b - g, seed=s))
        for a in ALGOS:                              # does seeding generalise?
            for init in ("random", "carbon"):
                for s in SEEDS:
                    out.append(J(exp=exp, N=n, M=M, cap=cap, pen=pen, hard=hard,
                                 method=a, init=init, seed=s))
    return out


def jobs_for(exp):
    if exp == "E1":                                  # published model, soft penalty
        return grid("E1")
    if exp == "E4":                                  # hard deadline constraint
        return grid("E4", hard=True)
    if exp == "E20":            # capacity as a FEASIBILITY constraint (correct model)
        return grid("E20", hard=True, pen=10.0)
    if exp == "E21":            # same, soft deadlines, to separate the two constraints
        return grid("E21", hard=False, pen=10.0)
    if exp == "E3":
        return ablation("E3", False, [500, 1500, 3000])
    if exp == "E22":            # ablation under the feasibility constraint
        return ablation("E22", False, [1000, 3000], pen=10.0)
    if exp == "E2":                                  # power models
        out = []
        for n, M, pw in itertools.product([500, 1500, 3000], MS,
                                          ["linear", "cubic", "piecewise"]):
            for m in DET:
                out.append(J(exp=exp, N=n, M=M, power=pw, method=m, seed=0, init="-"))
            for a in ALGOS:
                for s in SEEDS:
                    out.append(J(exp=exp, N=n, M=M, power=pw, method=a,
                                 init="random", seed=s))
            for s in SEEDS:
                out.append(J(exp=exp, N=n, M=M, power=pw, method="WOA",
                             init="carbon", seed=s))
        return out
    if exp == "E5":                                  # weights + seeded fraction
        GRID = [(0.4, 0.3, 0.3), (1.0, 0.0, 0.0), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2),
                (0.34, 0.33, 0.33), (0.2, 0.4, 0.4), (0.2, 0.6, 0.2),
                (0.2, 0.2, 0.6), (0.1, 0.8, 0.1), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)]
        out = []
        for n, M, cap in itertools.product([1000, 3000], MS, [False, True]):
            for (a, b, g) in GRID:
                for init in ("random", "carbon"):
                    for s in SEEDS:
                        out.append(J(exp=exp, N=n, M=M, cap=cap, alpha=a, beta=b,
                                     gamma=g, method="WOA", init=init, seed=s))
            for frac in (0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 1.0):
                for s in SEEDS:
                    out.append(J(exp=exp, N=n, M=M, cap=cap, method="WOA",
                                 init="carbon", seed_frac=frac, seed=s))
        return out
    if exp == "E7":                                  # workload subsets + NASA arrivals
        out = []
        for n, M in itertools.product([500, 1500, 3000], MS):
            for sub in (0, 1, 2, 3):
                out.append(J(exp=exp, N=n, M=M, subset=sub,
                             method="Greedy(EDF+greenest)", seed=0, init="-"))
                for init in ("random", "carbon"):
                    for s in SEEDS[:10]:
                        out.append(J(exp=exp, N=n, M=M, subset=sub, method="WOA",
                                     init=init, seed=s))
            out.append(J(exp=exp, wl="nasa", N=n, M=M,
                         method="Greedy(EDF+greenest)", seed=0, init="-"))
            for a in ALGOS:
                for s in SEEDS[:10]:
                    out.append(J(exp=exp, wl="nasa", N=n, M=M, method=a,
                                 init="random", seed=s))
            for s in SEEDS[:10]:
                out.append(J(exp=exp, wl="nasa", N=n, M=M, method="WOA",
                             init="carbon", seed=s))
        return out
    if exp == "E23":            # minimum active-host constraint (workload-determined)
        out = []
        for n, M, mm in itertools.product(NS_MAIN, MS, [0, "auto", 0.25, 0.5]):
            for m in DET:
                out.append(J(exp=exp, N=n, M=M, mmin=mm, hard=True, pen=10.0,
                             method=m, seed=0, init="-"))
            for m in B.POLICIES:
                out.append(J(exp=exp, N=n, M=M, mmin=mm, hard=True, pen=10.0,
                             method=m, seed=0, init="-"))
            for init in ("random", "carbon"):
                for s_ in SEEDS[:15]:
                    out.append(J(exp=exp, N=n, M=M, mmin=mm, hard=True, pen=10.0,
                                 method="WOA", init=init, seed=s_))
        return out
    if exp == "E24":            # modern published schedulers on the full grid
        out = []
        for n, M in itertools.product(NS_MAIN, MS):
            for hard, pen in ((True, 10.0), (False, 0.0)):
                for m in list(B.POLICIES) + list(DET):
                    out.append(J(exp=exp, N=n, M=M, hard=hard, pen=pen,
                                 method=m, seed=0, init="-"))
                for s_ in SEEDS:
                    out.append(J(exp=exp, N=n, M=M, hard=hard, pen=pen,
                                 method="WOA", init="carbon", seed=s_))
        return out
    if exp == "E25":            # runtime/memory at the task counts reviewers named
        out = []
        for n, M in itertools.product([50, 100, 200, 300], MS):
            for m in DET:
                out.append(J(exp=exp, N=n, M=M, method=m, seed=0, init="-"))
            for a in ALGOS:
                for s_ in SEEDS:
                    out.append(J(exp=exp, N=n, M=M, method=a, init="random", seed=s_))
            for s_ in SEEDS:
                out.append(J(exp=exp, N=n, M=M, method="WOA", init="carbon", seed=s_))
        return out
    if exp == "CONV":           # convergence curves (per-epoch global best fitness)
        out = []
        for n, M in itertools.product([500, 1500, 3000], MS):
            for a in ALGOS:
                for s_ in SEEDS[:10]:
                    out.append(J(exp=exp, N=n, M=M, method=a, init="random",
                                 seed=s_, curve=True))
            for s_ in SEEDS[:10]:
                out.append(J(exp=exp, N=n, M=M, method="WOA", init="carbon",
                             seed=s_, curve=True))
        return out
    if exp == "E26":            # multiple carbon regions x multiple real windows
        out = []
        regions = [r for r in ("UK", "IE", "NI", "US-CAL", "US-MIDA")
                   if os.path.exists(os.path.join(
                       os.path.dirname(OUT), "data", "carbon", K.REGIONS[r]))]
        print("   regions available: %s" % ", ".join(regions), flush=True)
        for reg, win, M in itertools.product(regions, range(6), MS):
            base = dict(exp=exp, N=1500, M=M, region=reg, window=win,
                        hard=True, pen=10.0)
            for m in list(DET) + list(B.POLICIES):
                out.append(J(method=m, seed=0, init="-", **base))
            for a in ALGOS:
                for s_ in SEEDS[:10]:
                    out.append(J(method=a, init="random", seed=s_, **base))
            for s_ in SEEDS[:10]:
                out.append(J(method="WOA", init="carbon", seed=s_, **base))
        return out
    if exp == "E29":            # heterogeneous fleets x startup/shutdown overheads
        out = []
        for n, M, fl, su in itertools.product([500, 1500, 3000], MS,
                                              ["uniform", "mixed", "skewed"],
                                              ["none", "low", "high"]):
            base = dict(exp=exp, N=n, M=M, fleet=fl, startup=su, hard=True, pen=10.0)
            for m in list(DET) + list(B.POLICIES):
                out.append(J(method=m, seed=0, init="-", **base))
            for init in ("random", "carbon"):
                for s_ in SEEDS[:10]:
                    out.append(J(method="WOA", init=init, seed=s_, **base))
        return out
    if exp == "E28":            # US Mid-Atlantic: a near-flat carbon signal (1.78x
                                # max/min vs 12.3x for the UK) -- the boundary case
                                # for temporal shifting, added after E26 was launched
        out = []
        for win, M in itertools.product(range(6), MS):
            base = dict(exp=exp, N=1500, M=M, region="US-MIDA", window=win,
                        hard=True, pen=10.0)
            for m in list(DET) + list(B.POLICIES):
                out.append(J(method=m, seed=0, init="-", **base))
            for a in ALGOS:
                for s_ in SEEDS[:10]:
                    out.append(J(method=a, init="random", seed=s_, **base))
            for s_ in SEEDS[:10]:
                out.append(J(method="WOA", init="carbon", seed=s_, **base))
        return out
    if exp == "E27":            # is the single published UK window representative?
        out = []
        nwin = len(K.carbon_windows("UK"))
        for win, M in itertools.product(range(nwin), MS):
            base = dict(exp=exp, N=1500, M=M, region="UK", window=win,
                        hard=True, pen=10.0)
            for m in list(DET) + list(B.POLICIES):
                out.append(J(method=m, seed=0, init="-", **base))
            for init in ("random", "carbon"):
                for s_ in SEEDS[:10]:
                    out.append(J(method="WOA", init=init, seed=s_, **base))
        return out
    if exp == "GABUG":                               # measure the OriginalGA defect
        out = []
        for n, M in itertools.product([500, 1500, 3000], MS):
            for m in ("GA", "GA_OriginalGA"):
                for s in SEEDS[:10]:
                    out.append(J(exp=exp, N=n, M=M, method=m, init="random", seed=s))
        return out
    raise ValueError("unknown experiment " + exp)


if __name__ == "__main__":
    for exp in sys.argv[1:]:
        path = os.path.join(OUT, "raw_%s.csv" % exp)
        if os.path.exists(path):
            print("%s already done, skipping" % exp, flush=True)
            continue
        js = jobs_for(exp)
        print("%s: %d runs" % (exp, len(js)), flush=True)
        t0 = time.time()
        rows = []
        with Pool(12) as p:
            for i, r in enumerate(p.imap_unordered(one, js, chunksize=8), 1):
                rows.append(r)
                if i % 1000 == 0:
                    print("   %d/%d  (%.0fs)" % (i, len(js), time.time() - t0), flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print("%s done: %d rows in %.0fs -> raw_%s.csv"
              % (exp, len(rows), time.time() - t0, exp), flush=True)

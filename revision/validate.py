"""Validation gate for core.py. Everything downstream depends on this passing.

  A. Vectorised evaluate() is numerically identical to the ORIGINAL dict-based
     evaluate() from week4_full_comparison.py.
  B. Rebuilding the PUBLISHED setup through core.py reproduces the published
     Table 4 numbers exactly.
  C. Power models and initialisation routines behave correctly.
  D. cap=True makes M a real limit; cap=False leaves the published model unchanged.
"""
import math, time, sys
import numpy as np
import core as K

CI, PRICE, H = K.load_carbon()
FAIL = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def ref_evaluate(tasks, M):
    """Faithful copy of the ORIGINAL dict-based evaluate() (week4_full_comparison.py)."""
    dur, u = tasks["dur"], tasks["u"]
    deadline = tasks["deadline"]
    N = len(dur)
    C, PI, PM, SH = 1.0, 100.0, 250.0, 0.5

    def ev(starts, consolidate=True):
        load, count = {}, {}
        for i in range(N):
            for k in range(starts[i], starts[i] + dur[i]):
                if k < H:
                    load[k] = load.get(k, 0.0) + u[i]
                    count[k] = count.get(k, 0) + 1
        carbon_g = cost = energy = 0.0
        util, overload, total_load, finish_max = [], 0.0, 0.0, 0
        for i in range(N):
            finish_max = max(finish_max, starts[i] + dur[i])
        viol = sum(1 for i in range(N) if starts[i] + dur[i] > deadline[i])
        for k, ld in load.items():
            active = (count[k] if not consolidate else max(1, math.ceil(ld / C)))
            p = active * PI + (PM - PI) * ld
            energy += p * SH / 1000.0
            carbon_g += p * SH / 1000.0 * CI[k]
            cost += p * SH / 1000.0 * PRICE[k]
            util.append(ld / (active * C))
            overload += max(0.0, ld - M)
            total_load += ld
        return {"Carbon_kgCO2": carbon_g / 1000.0, "Energy_kWh": energy, "Cost_GBP": cost,
                "SLA_%": 100.0 * viol / N, "Makespan_h": finish_max * SH,
                "Util_%": 100.0 * float(np.mean(util)) if util else 0.0,
                "Overload_%": 100.0 * overload / total_load if total_load else 0.0}
    return ev


build_published = K.build_published   # moved into core.py; re-exported here


# ============================================================== A
print("\nA. Vectorised evaluate() vs the original dict-based evaluate()")
rng = np.random.default_rng(7)
worst = 0.0
for n in (60, 200, 500, 1500):
    tasks = K.build_google(n, H)
    env = K.Env(tasks, CI, PRICE, H, M=10, power="linear")
    ref = ref_evaluate(tasks, M=10)
    for _ in range(6):
        starts = env.decode(rng.uniform(0, 1, n))
        for consol in (True, False):
            a, b = env.evaluate(starts, consol), ref(starts, consol)
            for k in ("Carbon_kgCO2", "Energy_kWh", "Cost_GBP", "SLA_%",
                      "Makespan_h", "Util_%", "Overload_%"):
                worst = max(worst, abs(a[k] - b[k]) / max(1e-12, abs(b[k])))
check("identical on all 7 metrics, 4 sizes x 6 schedules x 2 modes", worst < 1e-9,
      "worst relative difference = %.3e" % worst)

# ============================================================== B
print("\nB. Reproducing the PUBLISHED Table 4 through core.py")
t0 = build_published(60)
env0 = K.Env(t0, CI, PRICE, H, M=None, power="linear")     # M derived, as published
check("derived host count M == 5", env0.M == 5, "got M=%d, peak=%.2f" % (env0.M, env0.peak))
check("all 60 arrivals collapse to slot 0 (the shipped bug)",
      np.unique(t0["earliest"]).size == 1 and t0["earliest"][0] == 0,
      "distinct arrival slots = %d" % np.unique(t0["earliest"]).size)
check("consolidation-only reduction == 80.5", abs(env0.cred(env0.cons) - 80.5) < 0.05,
      "got %.2f" % env0.cred(env0.cons))
g = env0.evaluate(env0.greedy_starts(), consolidate=True)
check("carbon-aware greedy == 82.56", abs(env0.cred(g) - 82.56) < 0.05,
      "got %.2f" % env0.cred(g))
check("greedy utilisation == 75.2", abs(g["Util_%"] - 75.2) < 0.1, "got %.1f" % g["Util_%"])
check("naive FIFO energy == 5.03 kWh", abs(env0.base["Energy_kWh"] - 5.03) < 0.01,
      "got %.3f" % env0.base["Energy_kWh"])

# ============================================================== C
print("\nC. Power models and initialisation")
one = np.array([1.0])
for nm, f in K.POWER_MODELS.items():
    check("%-9s endpoints 100 W at u=0, 250 W at u=1" % nm,
          abs(f(one, np.array([0.0]))[0] - 100.0) < 1e-9 and
          abs(f(one, np.array([1.0]))[0] - 250.0) < 1e-9)
us = np.linspace(0, 1, 21); ones = np.ones_like(us)
for nm, f in K.POWER_MODELS.items():
    check("%-9s monotone non-decreasing" % nm, bool(np.all(np.diff(f(ones, us)) >= -1e-9)))
check("cubic is convex (below linear inside)",
      bool(np.all(K.p_cubic(ones, us)[1:-1] <= K.p_linear(ones, us)[1:-1] + 1e-9)))
check("piecewise is concave (above linear inside)",
      bool(np.all(K.p_piecewise(ones, us)[1:-1] >= K.p_linear(ones, us)[1:-1] - 1e-9)))

envh = K.Env(K.build_google(300, H), CI, PRICE, H, M=10, hard=True)
check("greedy is deadline-feasible", bool(np.all(envh.greedy_starts() + envh.dur <= envh.deadline)))
check("hard decode always deadline-feasible (200 random x)",
      all(np.all(envh.decode(rng.uniform(0, 1, envh.N)) + envh.dur <= envh.deadline)
          for _ in range(200)))
envs = K.Env(K.build_google(300, H), CI, PRICE, H, M=10, hard=False)
check("soft decode CAN violate deadlines (it is a penalty, not a constraint)",
      any((envs.decode(rng.uniform(0, 1, envs.N)) + envs.dur > envs.deadline).any()
          for _ in range(50)))
for nm, fn in (("random", envh.pop_random), ("improved", envh.pop_improved),
               ("carbon", envh.pop_carbon)):
    P = fn(40, np.random.default_rng(1))
    check("pop_%-8s shape (40,N), values in [0,1]" % nm,
          P.shape == (40, envh.N) and P.min() >= -1e-12 and P.max() <= 1 + 1e-12)
gx = envh.x_from_starts(envh.greedy_starts())
check("carbon seed round-trips to the greedy schedule",
      bool(np.array_equal(envh.decode(gx), envh.greedy_starts())))
check("improved init carries NO carbon information",
      not np.allclose(envh.pop_improved(40, np.random.default_rng(1))[0], gx))

# ============================================================== D
print("\nD. Capacity cap")
for n in (1000, 3000):
    r = {}
    for M in (10, 20):
        e = K.Env(K.build_google(n, H), CI, PRICE, H, M=M, cap=True)
        r[M] = (e.cred(e.cons), e.cons["Overload_%"], e.base["Overload_%"])
    check("cap=True: N=%d, M=10 vs M=20 give DIFFERENT results" % n,
          abs(r[10][0] - r[20][0]) > 1.0,
          "consolidation reduction %.2f vs %.2f" % (r[10][0], r[20][0]))
    check("cap=True: N=%d, M=10 FIFO overload is non-zero (M binds)" % n, r[10][2] > 0,
          "FIFO overload %.2f%%" % r[10][2])
e_nc = K.Env(K.build_google(1000, H), CI, PRICE, H, M=10, cap=False)
e_c = K.Env(K.build_google(1000, H), CI, PRICE, H, M=10, cap=True)
check("cap=False leaves the published model unchanged (differs from cap=True)",
      abs(e_nc.cred(e_nc.cons) - e_c.cred(e_c.cons)) > 1.0,
      "cap=False %.2f vs cap=True %.2f" % (e_nc.cred(e_nc.cons), e_c.cred(e_c.cons)))

print("\nE. Speed")
for n in (500, 1500, 3000):
    env = K.Env(K.build_google(n, H), CI, PRICE, H, M=10)
    x = rng.uniform(0, 1, n)
    t = time.time()
    for _ in range(200):
        env.fitness(x)
    per = (time.time() - t) / 200
    print("    N=%-5d %.6fs per evaluation -> %.2fs per 4840-eval run" % (n, per, per * 4840))

print("\n" + "=" * 72)
if FAIL:
    print("VALIDATION FAILED: %d check(s)" % len(FAIL))
    for f in FAIL:
        print("   - " + f)
    sys.exit(1)
print("ALL CHECKS PASSED - core.py is safe to build experiments on")

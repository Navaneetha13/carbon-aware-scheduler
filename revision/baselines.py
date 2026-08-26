"""Modern published scheduler baselines (Reviewer 3 #2, "one modern scheduler").

Two policies, both taken from published, peer-reviewed systems and reimplemented
as scheduling policies inside the same simulator, on the same traces, with the
same metrics. Neither is a metaheuristic and neither uses the fitness function,
so they are independent of CA-WOA's machinery.

  vcc_starts   -- Virtual Capacity Curves, the load-shaping policy of Google's
                  production carbon-aware scheduler (Radovanovic et al.,
                  "Carbon-Aware Computing for Datacenters", IEEE Transactions on
                  Power Systems 38(2), 2023). Per-slot compute capacity is
                  throttled as a function of carbon intensity; work is admitted
                  earliest-deadline-first into whatever capacity remains. Unlike
                  the greedy EDF+greenest rule this respects a per-slot ceiling by
                  construction, so it is a genuinely competitive baseline in the
                  capacitated regime.

  thresh_starts -- Threshold-based temporal shifting (Wiesner et al., "Let's Wait
                  Awhile: How Temporal Workload Shifting Can Reduce Carbon
                  Emissions in the Cloud", ACM/IFIP Middleware 2021). A task is
                  deferred while carbon intensity sits above a percentile of the
                  signal, subject to its deadline and the deferral limit.

Both return a start-slot vector, so `env.evaluate(starts, consolidate=True)`
scores them identically to every other method. This module only READS Env; it
does not modify core.py.
"""
import numpy as np


def _order(env):
    """Earliest-deadline-first, ties broken by earliest release then longest job."""
    return np.lexsort((-env.dur, env.earliest, env.deadline))


def _feasible_window(env, i):
    """Inclusive [lo, hi] start slots for task i, honouring release, deferral
    limit, horizon and (always, for these baselines) the deadline."""
    lo = int(env.earliest[i])
    hi = int(min(lo + env.room[i], env.deadline[i] - env.dur[i], env.H - env.dur[i]))
    return lo, max(lo, hi)


def vcc_starts(env, floor=0.35, M=None):
    """Virtual Capacity Curves (Google Carbon-Intelligent Computing).

    Capacity available in slot k is scaled down when carbon intensity is high:

        cap_k = M * C * (floor + (1 - floor) * (1 - ci_norm_k))

    where ci_norm linearly normalises CI over the horizon, so the greenest slot
    keeps full capacity and the dirtiest keeps `floor` of it. Tasks are then
    placed EDF-first into the earliest deadline-feasible slot whose remaining
    virtual capacity admits them; if none does, the task goes to the slot with the
    most remaining capacity in its window (the policy degrades rather than
    dropping work, which keeps it comparable to the other methods).
    """
    M = env.M if M is None else M
    ci = np.asarray(env.CI, float)[:env.H]
    span = ci.max() - ci.min()
    ci_norm = np.zeros_like(ci) if span <= 0 else (ci - ci.min()) / span
    cap = M * env.C * (floor + (1.0 - floor) * (1.0 - ci_norm))

    used = np.zeros(env.H)
    starts = env.earliest.copy()
    for i in _order(env):
        lo, hi = _feasible_window(env, i)
        d, u = int(env.dur[i]), float(env.u[i])
        best_s, best_slack = None, -np.inf
        for s in range(lo, hi + 1):
            sl = float(np.min(cap[s:s + d] - used[s:s + d]))
            if sl >= u:                      # fits inside the virtual capacity
                best_s = s
                break
            if sl > best_slack:              # remember the least-bad fallback
                best_slack, best_s = sl, s
        starts[i] = best_s
        used[best_s:best_s + d] += u
    return starts


def thresh_starts(env, pct=40.0):
    """Threshold-based temporal shifting ("Let's Wait Awhile").

    Defer to the first slot in the task's feasible window whose carbon intensity
    is at or below the `pct`-th percentile of the horizon; if the window contains
    no such slot, use its greenest slot. No capacity awareness -- this is the
    pure temporal-shifting policy, included to show what carbon awareness alone
    buys without any capacity reasoning.
    """
    ci = np.asarray(env.CI, float)[:env.H]
    thr = float(np.percentile(ci, pct))
    starts = env.earliest.copy()
    for i in range(env.N):
        lo, hi = _feasible_window(env, i)
        d = int(env.dur[i])
        win = np.arange(lo, hi + 1)
        mean_ci = np.array([ci[s:s + d].mean() for s in win])
        ok = np.flatnonzero(mean_ci <= thr)
        starts[i] = int(win[ok[0]] if ok.size else win[int(np.argmin(mean_ci))])
    return starts


POLICIES = {"VCC(Google)": vcc_starts, "Threshold(WaitAwhile)": thresh_starts}


if __name__ == "__main__":                    # smoke test against the live model
    import core as K
    print("%-24s %8s %9s %9s %8s" % ("method", "carbon%", "overload%", "feasible", "SLA%"))
    for n, M in ((1000, 10), (3000, 10), (3000, 20)):
        env = K.make_env(n=n, M=M, hard=True, pen=10.0)
        print("-- N=%d M=%d  (peak demand %.1f)" % (n, M, env.peak))
        rows = {"FIFO": env.fifo, "Greedy(EDF+greenest)": env.greedy_starts()}
        rows.update({k: f(env) for k, f in POLICIES.items()})
        for name, st in rows.items():
            mm = env.evaluate(np.asarray(st, int), consolidate=True)
            print("%-24s %8.2f %9.2f %9s %8.4f"
                  % (name, env.cred(mm), mm["Overload_%"], mm["Feasible"], mm["SLA_%"]))

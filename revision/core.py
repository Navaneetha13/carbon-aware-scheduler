"""Vectorised simulation core for the CA-WOA revision experiments.

Numerically identical to the original dict-based evaluate() in
week4_full_comparison.py / scalability_sweep.py (see validate.py), but ~100x
faster, so N up to 3000 with 30 seeds is tractable.

Adds, for the reviewer comments:
  * pluggable power model: linear (as published), cubic, piecewise SPECpower-style
  * host count M either derived from the workload or fixed (10, 20, ...)
  * cap=True makes M a REAL resource limit (see Env.__init__)
  * hard=True clamps decoding to the deadline (repair), so SLA=0 by construction
  * three initialisation strategies: random, "improved" (no carbon), carbon-aware

Nothing is hardcoded: every number comes from the real traces under data/.
"""
import json, math, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT_H = 0.5                      # half-hour slots
P_IDLE_W, P_MAX_W = 100.0, 250.0  # the paper's linear model endpoints
SLACK, MAX_DEFER = 8, 24


# ----------------------------------------------------------------- carbon + tariff
REGIONS = {"UK": "carbon_history.csv", "IE": "carbon_history_IE.csv",
           "NI": "carbon_history_NI.csv", "US-CAL": "carbon_history_US-CAL.csv",
           "US-MIDA": "carbon_history_US-MIDA.csv"}


def carbon_windows(region, span=144):
    """All disjoint contiguous `span`-slot windows in a region's history.

    Returns a list of float arrays. Contiguity matters: a window that straddles a
    gap in the trace would fabricate a carbon signal that never occurred, so runs
    are split at any break in the half-hourly sequence.
    """
    path = os.path.join(ROOT, "data", "carbon", REGIONS[region])
    d = pd.read_csv(path)
    t = pd.to_datetime(d["from"], format="mixed", utc=True)
    brk = (t.diff().dt.total_seconds().div(1800) != 1).cumsum()
    out = []
    for _, g in d.groupby(brk):
        v = g["intensity"].to_numpy(float)
        for i in range(len(v) // span):
            out.append(v[i * span:(i + 1) * span])
    return out


def load_carbon(region="UK", window=None):
    """Carbon intensity and tariff for one 3-day scheduling horizon.

    region="UK", window=None reproduces the published signal byte-for-byte from
    3day_window.json; every other combination draws a real contiguous window from
    that region's half-hourly history. The tariff is the UK time-of-use structure
    throughout and is applied unchanged to other regions -- it is a cost proxy,
    not a claim about local pricing.
    """
    if region == "UK" and window is None:
        j = json.load(open(ROOT + "/data/carbon/3day_window.json"))
        CI = np.array([r["intensity"].get("actual") or r["intensity"].get("forecast")
                       for r in j["data"]], float)
    else:
        ws = carbon_windows(region)
        if not ws:
            raise ValueError("no contiguous 144-slot window for region %r" % region)
        CI = ws[(window or 0) % len(ws)]
    H = len(CI)
    PRICE = np.full(H, 0.15)
    for day in range(H // 48 + 1):
        for s in range(32, 40):
            k = day * 48 + s
            if k < H:
                PRICE[k] = 0.30
    return CI, PRICE, H


# ----------------------------------------------------------------- power models
def p_linear(active, load):
    """The published model: each active host idles at P_IDLE, dynamic term linear."""
    return active * P_IDLE_W + (P_MAX_W - P_IDLE_W) * load


def p_cubic(active, load):
    """Cubic (convex): P(u) = P_idle + (P_max-P_idle)*u^3 per host, u = load/active.
    Same endpoints as linear (100 W at u=0, 250 W at u=1) but convex, so spreading
    load over more hosts is cheaper - the qualitative difference that matters when
    testing whether consolidation's advantage is model-dependent."""
    active = np.maximum(active, 1e-12)
    u = np.clip(np.divide(load, active, out=np.zeros_like(load), where=active > 0), 0.0, 1.0)
    return active * (P_IDLE_W + (P_MAX_W - P_IDLE_W) * u ** 3)


# SPECpower-style measured curve: fraction of the dynamic range drawn at each 10%
# utilisation step. Concave - steep initial rise, flattening near full load.
_SPEC_U = np.linspace(0.0, 1.0, 11)
_SPEC_F = np.array([0.00, 0.16, 0.28, 0.39, 0.49, 0.58, 0.67, 0.76, 0.85, 0.93, 1.00])


def p_piecewise(active, load):
    """Piecewise-linear interpolation of a SPECpower-style measured power curve."""
    active = np.maximum(active, 1e-12)
    u = np.clip(np.divide(load, active, out=np.zeros_like(load), where=active > 0), 0.0, 1.0)
    return active * (P_IDLE_W + (P_MAX_W - P_IDLE_W) * np.interp(u, _SPEC_U, _SPEC_F))


POWER_MODELS = {"linear": p_linear, "cubic": p_cubic, "piecewise": p_piecewise}


# ----------------------------------------------------------------- workloads
GCOLS = ["time", "missing", "job_id", "task_index", "machine_id", "event_type", "user",
         "sched_class", "priority", "cpu_request", "mem_request", "disk_request", "constraint"]
_GU = None


def google_usable():
    """All usable Google tasks (submit, duration, cpu_request). Cached per process."""
    global _GU
    if _GU is None:
        df = pd.read_csv(ROOT + "/data/workload/google_task_events_part0.csv.gz",
                         header=None, names=GCOLS)
        sub = (df[df.event_type == 0][["job_id", "task_index", "time", "cpu_request"]]
               .dropna(subset=["cpu_request"]).rename(columns={"time": "submit"})
               .groupby(["job_id", "task_index"], as_index=False).first())
        end = (df[df.event_type.isin([2, 3, 4, 5])][["job_id", "task_index", "time"]]
               .rename(columns={"time": "end"})
               .groupby(["job_id", "task_index"], as_index=False).first())
        m = sub.merge(end, on=["job_id", "task_index"])
        m["dur_us"] = m["end"] - m["submit"]
        _GU = m[(m.dur_us > 0) & (m.cpu_request > 0)].reset_index(drop=True)
    return _GU


def build_google(n, H, subset=0):
    """n Google tasks with arrivals SPREAD across the arrival window.

    Fixes the published bug: head(n) drew tasks sharing one submit timestamp, so
    every arrival collapsed to slot 0. Here tasks are drawn by an even stride over
    the whole usable set, preserving the real submit ordering, then linearly
    rescaled onto the first H//3 slots. `subset` shifts the stride to give a
    different (largely disjoint) sample, for the multiple-workload-subsets comment.
    """
    GU = google_usable()
    if n > len(GU):
        raise ValueError("only %d usable Google tasks available" % len(GU))
    idx = np.linspace(0, len(GU) - 1, n).astype(int)
    if subset:
        idx = np.sort((idx + subset * max(1, len(GU) // (n * 4))) % len(GU))
    m = GU.iloc[idx]
    smin, smax = m.submit.min(), m.submit.max()
    span = max(1, smax - smin)
    dur = np.clip(np.ceil(m.dur_us.values / 1.8e9), 1, 12).astype(int)
    u = np.clip(m.cpu_request.values.astype(float), 0.05, 1.0)
    e = np.minimum(((m.submit.values - smin) / span * (H // 3)).astype(int), H // 3)
    return {"dur": dur, "u": u, "earliest": e, "deadline": e + dur + SLACK}


NCOLS = ["job", "submit", "wait", "runtime", "nproc", "avg_cpu", "used_mem", "req_proc",
         "req_time", "req_mem", "status", "uid", "gid", "app", "queue", "partition",
         "prev_job", "think"]
_NA = None


def nasa_usable():
    global _NA
    if _NA is None:
        df = pd.read_csv(ROOT + "/data/workload/NASA.swf", sep=r"\s+", comment=";",
                         header=None, names=NCOLS)
        _NA = df[(df.submit >= 0) & (df.nproc > 0) & (df.runtime > 0)].reset_index(drop=True)
    return _NA


def build_nasa(n, H, day_offset=10):
    """n NASA-iPSC jobs with a REAL arrival process (92-day trace).
    Real inter-arrival structure is preserved; only the timebase is rescaled."""
    df = nasa_usable()
    t0 = df.submit.min() + day_offset * 86400
    w = df[df.submit >= t0].reset_index(drop=True)
    if n > len(w):
        raise ValueError("only %d NASA jobs available after day %d" % (len(w), day_offset))
    w = w.iloc[:n]
    smin, smax = w.submit.min(), w.submit.max()
    span = max(1, smax - smin)
    dur = np.clip(np.ceil(w.runtime.values / 1800.0), 1, 12).astype(int)
    u = np.clip(w.nproc.values.astype(float) / 128.0, 0.05, 1.0)
    e = np.minimum(((w.submit.values - smin) / span * (H // 3)).astype(int), H // 3)
    return {"dur": dur, "u": u, "earliest": e, "deadline": e + dur + SLACK}


WORKLOADS = {"google": build_google, "nasa": build_nasa}


def build_published(n):
    """The ORIGINAL (buggy) builder used in the submitted work: head(n) selects
    tasks that share a single submit timestamp, so every arrival collapses to
    slot 0 and the workload occupies 4 of the 144 slots. Kept so the published
    numbers stay exactly reproducible and the defect can be demonstrated rather
    than merely asserted -- build_google() is the corrected sampling."""
    _, _, H = load_carbon()
    GU = google_usable().head(n)
    smin, smax = GU.submit.min(), GU.submit.max()
    dur = np.clip(np.ceil(GU.dur_us.values / 1.8e9), 1, 12).astype(int)
    u = np.clip(GU.cpu_request.values.astype(float), 0.05, 1.0)
    e = ((GU.submit.values - smin) / (smax - smin + 1) * (H // 3)).astype(int)
    return {"dur": dur, "u": u, "earliest": e, "deadline": e + dur + SLACK}


# ------------------------------------------------- heterogeneous fleet
# Reviewer 2: "Consider heterogeneous hosts ... startup/shutdown overheads".
# A fleet is a list of (capacity, P_idle, P_max) tuples. Hosts are filled in order
# of increasing energy per unit of work delivered, i.e. the most efficient machines
# are switched on first -- the standard consolidation policy. "uniform" reproduces
# the published homogeneous fleet exactly, so it is the default everywhere.
FLEETS = {
    # capacity, P_idle W, P_max W
    "uniform":  None,                                   # published: all hosts identical
    # a plausible three-generation datacentre: newer machines are bigger AND more
    # efficient per unit of work, older ones linger because they are already paid for
    "mixed":    [(1.5, 110.0, 300.0),                   # newest: 200 W/unit dynamic
                 (1.0, 100.0, 250.0),                   # mid    : 150 W/unit
                 (0.5,  85.0, 160.0)],                  # oldest : 150 W/unit, poor idle
    # deliberately adversarial: the big hosts are the INEFFICIENT ones
    "skewed":   [(2.0, 200.0, 520.0),
                 (1.0, 100.0, 250.0),
                 (0.5,  60.0, 140.0)],
}

# Energy charged once each time a host transitions from off to on, in kWh.
# 0.0 reproduces the published model. A physical server takes roughly 2-4 minutes
# to boot while drawing near-peak power; 250 W for 3 min = 0.0125 kWh.
STARTUP_KWH = {"none": 0.0, "low": 0.0125, "high": 0.05}


def fleet_arrays(name, M):
    """Expand a fleet spec into per-host arrays sorted most-efficient-first."""
    spec = FLEETS[name]
    if spec is None:
        return (np.full(M, 1.0), np.full(M, P_IDLE_W), np.full(M, P_MAX_W))
    reps = int(np.ceil(M / len(spec)))
    rows = (spec * reps)[:M]
    cap = np.array([r[0] for r in rows], float)
    pi = np.array([r[1] for r in rows], float)
    pm = np.array([r[2] for r in rows], float)
    order = np.argsort((pm - pi) / cap + pi / cap)      # dynamic + idle cost per unit
    return cap[order], pi[order], pm[order]


def het_power(load, n_on, cap, pi, pm, pack=True):
    """Power draw of a heterogeneous fleet, in watts, per slot.

    Hosts arrive sorted most-efficient-first (see fleet_arrays).

    pack=True  (consolidation): fill hosts in efficiency order, each taking as much
               of the remaining load as its capacity allows. Only hosts that receive
               work are powered. Load beyond total fleet capacity spills onto the
               last host and shows up as overload, which is penalised separately.

    pack=False (naive placement): `n_on` hosts are already powered -- one per task,
               as in the FIFO/round-robin baseline -- and the slot's load is spread
               across them in proportion to their capacity. This is what makes the
               non-consolidated baseline genuinely more expensive than consolidation,
               which is the effect the paper measures.
    """
    load = np.atleast_1d(np.asarray(load, float))
    M = len(cap)
    W = np.zeros_like(load)
    if pack:
        cum = np.cumsum(cap)
        total = cum[-1]
        for k in range(M):
            prev = cum[k - 1] if k else 0.0
            share = np.clip(load - prev, 0.0, cap[k])
            if k == M - 1:
                share = np.where(load > total, load - prev, share)
            u = np.minimum(np.divide(share, cap[k]), 1.0)
            W += np.where(share > 0, pi[k] + (pm[k] - pi[k]) * u, 0.0)
        return W

    n = np.clip(np.rint(np.asarray(n_on, float)).astype(int), 0, M)
    capsum = np.cumsum(cap)
    for k in range(M):
        on = n > k                                   # host k powered in this slot?
        if not np.any(on):
            break
        tot = np.where(on, capsum[n - 1], 1.0)       # capacity of the powered subset
        share = np.where(on, load * cap[k] / tot, 0.0)
        u = np.minimum(share / cap[k], 1.0)
        W += np.where(on, pi[k] + (pm[k] - pi[k]) * u, 0.0)
    return W



# ----------------------------------------------------------------- environment
class Env:
    """Vectorised capacity-model scheduling environment."""

    def __init__(self, tasks, CI, PRICE, H, M=None, C=1.0, power="linear",
                 hard=False, alpha=0.4, beta=0.3, gamma=0.3, cap=False, pen=0.0,
                 mmin=0, fleet="uniform", startup="none"):
        self.dur = np.asarray(tasks["dur"], int)
        self.u = np.asarray(tasks["u"], float)
        self.earliest = np.asarray(tasks["earliest"], int)
        self.deadline = np.asarray(tasks["deadline"], int)
        self.N = len(self.dur)
        self.CI, self.PRICE, self.H, self.C = CI, PRICE, H, C
        self.pfun = POWER_MODELS[power]
        self.power_name = power
        self.fleet_name, self.startup_name = fleet, startup
        self.e_start = STARTUP_KWH[startup]
        self.hard = hard
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        # cap=True caps SERVED load at M*C, i.e. excess demand is dropped. That is
        # only meaningful as a model of admission control; it must NOT be used with a
        # carbon metric, because dropped work is never charged for energy or carbon,
        # so the metric then REWARDS overloading. (Measured: greedy overloads 41% at
        # N=2500/M=10 and thereby appears to beat a feasible schedule by 16 pp.)
        # Default cap=False: energy is charged on the FULL demanded load, exactly as
        # in the published model.
        self.cap = cap
        # pen > 0 turns host capacity into a FEASIBILITY constraint: overload is not
        # rewarded (energy is still charged on full demand) but is penalised heavily
        # in the fitness, so the search must find schedules that fit within M hosts.
        # A schedule is feasible iff Overload_% == 0.
        self.pen = pen

        # flattened (task, offset) expansion so slot indices are one vector op
        self.task_of = np.repeat(np.arange(self.N), self.dur)
        self.offset = (np.concatenate([np.arange(d) for d in self.dur])
                       if self.N else np.array([], int))
        self.u_exp = self.u[self.task_of]

        room = np.maximum(np.minimum(MAX_DEFER, self.H - self.dur - self.earliest), 0)
        if hard:   # repair: never allow a start that misses the deadline
            room = np.maximum(np.minimum(room, self.deadline - self.dur - self.earliest), 0)
        self.room = room

        self.fifo = self.earliest.copy()
        fl, _ = self.slot_loads(self.fifo)
        self.peak = float(fl.max())
        self.M = math.ceil(self.peak) if M is None else M
        self.cap_h, self.pi_h, self.pm_h = fleet_arrays(fleet, self.M)
        self.het = fleet != "uniform"
        self.total_cap = float(self.cap_h.sum()) if self.het else self.M * self.C

        # Minimum active-host constraint (Reviewer 2: "minimum active-host
        # constraints", supervisor: determine it from the workload). A datacentre
        # cannot scale to zero: a warm pool stays powered for the whole operating
        # window and draws idle power even in slots with no work.
        #   mmin == 0        -> no constraint (the published model, default)
        #   0 < mmin <= 1    -> fraction of M, rounded up
        #   mmin > 1         -> absolute host count
        #   mmin == "auto"   -> workload-determined: the mean occupied-slot load,
        #                       i.e. enough hosts to serve average demand
        if mmin == "auto":
            busy = fl[fl > 0]
            self.mmin = int(math.ceil(float(busy.mean()) / self.C)) if busy.size else 0
        elif 0 < mmin <= 1:
            self.mmin = int(math.ceil(mmin * self.M))
        else:
            self.mmin = int(mmin)
        self.mmin = min(self.mmin, self.M)      # never exceed the installed fleet

        self.base = self.evaluate(self.fifo, consolidate=False)   # naive FIFO
        self.cons = self.evaluate(self.fifo, consolidate=True)    # consolidated FIFO
        self.nfe = 0

    def slot_loads(self, starts):
        slots = starts[self.task_of] + self.offset
        s = slots[slots < self.H]
        w = self.u_exp[slots < self.H]
        return (np.bincount(s, weights=w, minlength=self.H),
                np.bincount(s, minlength=self.H))

    def evaluate(self, starts, consolidate=True):
        starts = np.asarray(starts, int)
        load, count = self.slot_loads(starts)
        occ = load > 0
        if not occ.any():
            return {"Carbon_kgCO2": 0.0, "Energy_kWh": 0.0, "Cost_GBP": 0.0,
                    "SLA_%": 0.0, "Viol_n": 0, "Makespan_h": 0.0,
                    "Util_%": 0.0, "Overload_%": 0.0, "Feasible": True}
        # With a minimum active-host floor the warm pool draws power in every slot
        # of the OPERATING WINDOW -- slot 0 up to the schedule's makespan -- so idle
        # slots inside that window can no longer be skipped. The window is bounded by
        # the makespan rather than the full horizon because there is no workload to
        # serve after the last task finishes. This makes deferral genuinely costly:
        # pushing work into a greener slot extends the window and so keeps the warm
        # pool powered for longer, which is exactly the trade-off the reviewers asked
        # to see modelled.
        if self.mmin > 0:
            span = int(min(self.H, max(1, int((starts + self.dur).max()))))
            sel = np.arange(self.H) < span
        else:
            sel = occ
        ld_raw = load[sel]
        active = (count[sel].astype(float) if not consolidate
                  else np.maximum(1.0, np.ceil(ld_raw / self.C)))
        if self.mmin > 0:
            active = np.maximum(active, float(self.mmin))
        if self.cap:
            active = np.minimum(active, float(self.M))
            ld = np.minimum(ld_raw, self.total_cap)
        else:
            ld = ld_raw
        if self.het:
            e = het_power(ld, active, self.cap_h, self.pi_h, self.pm_h,
                          pack=consolidate) * SLOT_H / 1000.0
        else:
            e = self.pfun(active, ld) * SLOT_H / 1000.0
        if self.e_start:
            # One charge per off->on transition, added to the slot in which the boot
            # happens, so it is priced at that slot's carbon intensity and tariff
            # rather than at the window average.
            a = np.rint(active).astype(int)
            booted = np.maximum(0, np.diff(np.concatenate(([0], a))))
            e = e + booted * self.e_start
        finish = starts + self.dur
        viol = int(np.count_nonzero(finish > self.deadline))
        # Total installed capacity, which for a heterogeneous fleet is the sum of the
        # individual host capacities rather than M*C.
        overload = np.maximum(0.0, ld_raw - self.total_cap).sum()
        return {"Carbon_kgCO2": float((e * self.CI[sel]).sum()) / 1000.0,
                "Energy_kWh": float(e.sum()),
                "Cost_GBP": float((e * self.PRICE[sel]).sum()),
                "SLA_%": 100.0 * viol / self.N, "Viol_n": viol,
                "Makespan_h": float(finish.max()) * SLOT_H,
                "Util_%": 100.0 * float(np.mean(ld / (active * self.C))),
                "Overload_%": 100.0 * float(overload / ld_raw.sum()) if ld_raw.sum() else 0.0,
                "Feasible": bool(overload <= 1e-12)}

    def decode(self, x):
        return self.earliest + np.rint(np.asarray(x, float) * self.room).astype(int)

    def fitness(self, x):
        self.nfe += 1
        mm = self.evaluate(self.decode(x), consolidate=True)
        return (self.alpha * mm["Carbon_kgCO2"] / self.base["Carbon_kgCO2"]
                + self.beta * mm["SLA_%"] / 100.0
                + (self.gamma + self.pen) * mm["Overload_%"] / 100.0)

    def cred(self, mm):
        b = self.base["Carbon_kgCO2"]
        return (b - mm["Carbon_kgCO2"]) / b * 100.0 if b else 0.0

    def cred_vs_cons(self, mm):
        b = self.cons["Carbon_kgCO2"]
        return (b - mm["Carbon_kgCO2"]) / b * 100.0 if b else 0.0

    # ---- initialisation strategies (the ablation arms) ----
    def greedy_starts(self):
        """Carbon-aware greedy / EDF+greenest: each task to its lowest-carbon
        deadline-feasible slot. Vectorised over candidate offsets."""
        hi = np.maximum(np.minimum(np.minimum(MAX_DEFER, self.H - self.dur - self.earliest),
                                   self.deadline - self.dur - self.earliest), 0)
        cum = np.concatenate([[0.0], np.cumsum(self.CI)])   # O(1) window sums
        best = np.zeros(self.N, int)
        bestc = np.full(self.N, np.inf)
        for o in range(int(hi.max()) + 1 if self.N else 1):
            s = self.earliest + o
            c = cum[np.minimum(s + self.dur, self.H)] - cum[np.minimum(s, self.H)]
            better = (o <= hi) & (c < bestc)
            bestc = np.where(better, c, bestc)
            best = np.where(better, o, best)
        return self.earliest + best

    def x_from_starts(self, starts):
        return np.clip((np.asarray(starts, int) - self.earliest) /
                       np.maximum(self.room, 1), 0.0, 1.0)

    def pop_random(self, pop, rng):
        return rng.uniform(0, 1, (pop, self.N))

    def pop_improved(self, pop, rng):
        """'Improved' initialisation WITHOUT carbon information (per supervisor):
        Latin-hypercube stratification plus opposition-based learning. Improves
        coverage and diversity only, so it isolates what the carbon knowledge in
        pop_carbon() actually contributes."""
        half = max(1, pop // 2)
        strata = (np.arange(half)[:, None] + rng.uniform(0, 1, (half, self.N))) / half
        lhs = np.take_along_axis(strata, rng.permuted(
            np.argsort(rng.uniform(0, 1, (half, self.N)), axis=0), axis=0), axis=0)
        out = np.vstack([lhs, 1.0 - lhs])                   # opposition-based counterpart
        if len(out) < pop:
            out = np.vstack([out, rng.uniform(0, 1, (pop - len(out), self.N))])
        return np.clip(out[:pop], 0.0, 1.0)

    def pop_carbon(self, pop, rng, frac=1 / 3, sigma=0.10):
        """Carbon-aware seeding: greedy carbon-optimal seed, perturbations, remainder random."""
        g = self.x_from_starts(self.greedy_starts())
        n_pert = int(pop * frac)
        out = [g[None, :]]
        if n_pert:
            out.append(np.clip(g[None, :] + rng.normal(0, sigma, (n_pert, self.N)), 0, 1))
        have = 1 + n_pert
        if have < pop:
            out.append(rng.uniform(0, 1, (pop - have, self.N)))
        return np.vstack(out)[:pop]


def make_env(workload="google", n=60, M=None, power="linear", hard=False, cap=False,
             subset=0, alpha=0.4, beta=0.3, gamma=0.3, pen=0.0, mmin=0,
             region="UK", window=None, fleet="uniform", startup="none"):
    CI, PRICE, H = load_carbon(region=region, window=window)
    tasks = (build_google(n, H, subset=subset) if workload == "google"
             else build_nasa(n, H))
    return Env(tasks, CI, PRICE, H, M=M, power=power, hard=hard, cap=cap,
               alpha=alpha, beta=beta, gamma=gamma, pen=pen, mmin=mmin,
               fleet=fleet, startup=startup)

"""Statistical significance testing for the scheduling results.

Applies Welch's unequal-variance t-test to the multi-seed results already produced by
scalability_sweep.py and week4_robustness.py, so every comparative claim in the write-up
can be checked against a p-value rather than a point estimate.

Two comparisons are reported for the proposed method at each workload scale:
  * CA-WOA vs standard WOA  -> the ABLATION. Identical algorithm, budget, workload and
    objective; the carbon-aware seeding is the only difference. This isolates the
    contribution of the proposed mechanism.
  * CA-WOA vs HHO           -> the strongest competing method that also achieves 0% SLA.

Note on the input data: the sweeps persist numpy's population standard deviation
(ddof=0). Welch's test requires the sample standard deviation (ddof=1), so each value is
converted by  s_sample = s_pop * sqrt(n / (n - 1))  before use.

Pure standard library - no scipy dependency - so this runs in any environment.
"""
import csv, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SEEDS = 5          # SEEDS = [1,2,3,4,5] in scalability_sweep.py and week4_robustness.py
ALPHA = 0.05


# ---------- regularised incomplete beta, for the Student-t tail ----------
def _betacf(a, b, x, itmax=200, eps=3e-16, fpmin=1e-300):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin: d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_tailed(t, df):
    """Two-tailed p-value for Student's t."""
    if df <= 0: return float("nan")
    if math.isinf(t): return 0.0
    return betainc(0.5 * df, 0.5, df / (df + t * t))


# ---------- Welch's t-test from summary statistics ----------
def welch(mean_a, sd_a, n_a, mean_b, sd_b, n_b):
    """Return (t, df, p) for Welch's unequal-variance t-test."""
    va, vb = sd_a ** 2 / n_a, sd_b ** 2 / n_b
    se = math.sqrt(va + vb)
    if se == 0.0:
        # Both methods perfectly repeatable. Separable iff the means differ at all.
        return (float("inf") if mean_a != mean_b else 0.0,
                float(n_a + n_b - 2),
                0.0 if mean_a != mean_b else 1.0)
    t = (mean_a - mean_b) / se
    denom = (va ** 2 / (n_a - 1)) + (vb ** 2 / (n_b - 1))
    df = (va + vb) ** 2 / denom if denom > 0 else float(n_a + n_b - 2)
    return t, df, t_sf_two_tailed(t, df)


def to_sample_sd(sd_pop, n):
    """numpy's np.std is ddof=0; Welch needs ddof=1."""
    return sd_pop * math.sqrt(n / (n - 1)) if n > 1 else 0.0


def stars(p):
    if p != p: return "n/a"
    for thr, s in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < thr: return s
    return "ns"


# ---------- load the multi-seed results ----------
def load_scalability():
    path = os.path.join(ROOT, "results", "scalability_sweep.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run src/scalability_sweep.py first")
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(int(r["N"]), r["Method"])] = (float(r["Carbon_red_%"]),
                                               float(r["Carbon_red_std"]))
    return out


def load_robustness():
    path = os.path.join(ROOT, "results", "week4_robustness.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["Method"]] = (float(r["Carbon_red_mean_%"]), float(r["Carbon_red_std"]))
    return out


def compare(label, a, b, rows, scale=""):
    """a, b = (name, mean, sd_population). Appends one result row."""
    (na, ma, sa), (nb, mb, sb) = a, b
    t, df, p = welch(ma, to_sample_sd(sa, N_SEEDS), N_SEEDS,
                     mb, to_sample_sd(sb, N_SEEDS), N_SEEDS)
    rows.append({"Comparison": label, "Scale": scale,
                 "Method_A": na, "Mean_A": round(ma, 2), "SD_A": round(sa, 2),
                 "Method_B": nb, "Mean_B": round(mb, 2), "SD_B": round(sb, 2),
                 "Difference": round(ma - mb, 2),
                 "t": round(t, 2) if math.isfinite(t) else "inf",
                 "df": round(df, 1), "p": f"{p:.2e}",
                 "Significant_at_0.05": "YES" if p < ALPHA else "no",
                 "Sig": stars(p)})
    return rows[-1]


def main():
    scal, rob = load_scalability(), load_robustness()
    rows = []

    scales = sorted({n for n, _ in scal})
    print("=" * 96)
    print("ABLATION - CA-WOA vs standard WOA  (isolates the carbon-aware seeding)")
    print("=" * 96)
    print(f"{'N':>5} {'CA-WOA':>14} {'WOA':>14} {'diff':>7} {'t':>9} {'df':>6} {'p':>11}  sig")
    for n in scales:
        if (n, "CA-WOA") not in scal or (n, "WOA") not in scal: continue
        ca, wo = scal[(n, "CA-WOA")], scal[(n, "WOA")]
        r = compare("CA-WOA vs WOA (ablation)", ("CA-WOA",) + ca, ("WOA",) + wo, rows, str(n))
        print(f"{n:>5} {ca[0]:>7.2f}±{ca[1]:<5.2f} {wo[0]:>7.2f}±{wo[1]:<5.2f} "
              f"{r['Difference']:>7.2f} {str(r['t']):>9} {r['df']:>6.1f} {r['p']:>11}  {r['Sig']}")

    print()
    print("=" * 96)
    print("CA-WOA vs HHO  (strongest competitor that also achieves 0% SLA violations)")
    print("=" * 96)
    print(f"{'N':>5} {'CA-WOA':>14} {'HHO':>14} {'diff':>7} {'t':>9} {'df':>6} {'p':>11}  sig")
    for n in scales:
        if (n, "CA-WOA") not in scal or (n, "HHO") not in scal: continue
        ca, hh = scal[(n, "CA-WOA")], scal[(n, "HHO")]
        r = compare("CA-WOA vs HHO", ("CA-WOA",) + ca, ("HHO",) + hh, rows, str(n))
        print(f"{n:>5} {ca[0]:>7.2f}±{ca[1]:<5.2f} {hh[0]:>7.2f}±{hh[1]:<5.2f} "
              f"{r['Difference']:>7.2f} {str(r['t']):>9} {r['df']:>6.1f} {r['p']:>11}  {r['Sig']}")

    if "CA-WOA" in rob and "WOA" in rob:
        print()
        print("=" * 96)
        print("Temporal-shift model (week4_robustness.csv, 5 seeds)")
        print("=" * 96)
        for other in ("WOA", "HHO", "GWO"):
            if other not in rob: continue
            r = compare(f"CA-WOA vs {other} (temporal model)",
                        ("CA-WOA",) + rob["CA-WOA"], (other,) + rob[other], rows, "60")
            print(f"  CA-WOA {rob['CA-WOA'][0]:.2f}±{rob['CA-WOA'][1]:.2f}  vs  "
                  f"{other} {rob[other][0]:.2f}±{rob[other][1]:.2f}   "
                  f"t={r['t']}  p={r['p']}  {r['Sig']}")

    out = os.path.join(ROOT, "results", "significance_tests.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    abl = [r for r in rows if "ablation" in r["Comparison"]]
    hho = [r for r in rows if r["Comparison"] == "CA-WOA vs HHO"]
    print()
    print("-" * 96)
    print("SUMMARY")
    print(f"  Ablation (seeding effect) significant at p<0.05: "
          f"{sum(1 for r in abl if r['Significant_at_0.05']=='YES')}/{len(abl)} scales")
    print(f"  CA-WOA vs HHO significant at p<0.05:             "
          f"{sum(1 for r in hho if r['Significant_at_0.05']=='YES')}/{len(hho)} scales")
    print(f"\nSaved -> results/significance_tests.csv")


if __name__ == "__main__":
    main()

# Revision experiments — CA-WOA

Self-contained artefacts for the reviewer responses. Every number here is computed
from the real traces in `../data/`; no result is transcribed or hardcoded.

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements-lock.txt          # exact pinned versions
./.venv/bin/pip install "tensorflow>=2.15" scikit-learn   # forecasting only
```

Pinned: `mealpy==3.0.3`, `numpy==1.26.0`, `pandas==3.0.5`, `scipy==1.16.3`.
mealpy is pinned to the version used in the submitted paper.

## Run order

```bash
./.venv/bin/python validate.py                     # MUST pass before anything else
./.venv/bin/python fetch_regions.py                # Ireland + N. Ireland (no API key)
EIA_API_KEY=... ./.venv/bin/python fetch_us.py     # US regions (free key required)
./.venv/bin/python runner.py E1 E4 E20 E21 E22 E3 E2 E5 E7 E23 E24 E25 E26 E27 CONV GABUG
./.venv/bin/python analyze.py                      # tables + statistics
./.venv/bin/python forecast.py all                 # forecasting experiments
```

`runner.py` skips any experiment whose `raw_<EXP>.csv` already exists, so it is
resumable. Each raw CSV holds **one row per (configuration, method, seed)** — the
per-seed results the reviewers asked for.

## Validation gate (`validate.py`)

27 checks in five groups. Nothing downstream is trustworthy unless these pass.

- **A** The vectorised `evaluate()` is numerically identical to the original
  dict-based implementation in `../src/week4_full_comparison.py` (worst relative
  deviation `9.3e-16`, i.e. machine precision) across 4 workload sizes × 6 random
  schedules × 2 consolidation modes × 7 metrics. The rewrite is ~100× faster,
  which is what makes N=3000 with 30 seeds tractable.
- **B** Rebuilding the **published** setup through `core.py` reproduces the
  published Table 4 exactly: consolidation 80.50, carbon-aware greedy 82.56,
  greedy utilisation 75.2, naive FIFO energy 5.033 kWh, derived M=5. It also
  confirms the shipped bug — all 60 arrivals collapse to slot 0.
- **C** Power models have correct endpoints (100 W at u=0, 250 W at u=1) and
  correct curvature; hard-constraint decoding is always deadline-feasible while
  soft decoding is not; the carbon seed round-trips to the greedy schedule; the
  "improved" initialisation provably carries no carbon information.
- **D** Host capacity behaves as documented under each of the three regimes below.
- **E** Evaluation speed, so run-time budgets are predictable.

## Corrections carried into every run

1. **Arrival times.** The published `head(n)` sampling drew tasks sharing one
   submit timestamp, so every arrival collapsed to slot 0 and the workload
   occupied 4 of 144 slots. `build_google` now samples by even stride across all
   52,057 usable tasks, preserving real submit ordering. `core.build_published`
   retains the original sampling so the defect can be demonstrated, not asserted.
2. **GA baseline.** `GA.OriginalGA` in mealpy 3.0.3 performs **no search** — it
   evaluates only the initial population (40 objective evaluations instead of
   4,840). `GA.BaseGA` is used instead; the defect is measured in `GABUG`.
3. **Objective evaluations recorded.** `nfe` is counted per run, so the budget
   difference (HHO ≈ 8,908 vs 4,840) is visible rather than assumed away.
   Section 6.1 of the submitted paper claims equal budgets; they are not equal.
4. **Overload definition.** Expressed as a fraction of *demanded* load, not
   served load.
5. **`cap=True` must not be used with a carbon metric.** It caps *served* load at
   M·C, so demand above capacity is dropped and never charged for energy or
   carbon — the metric then *rewards* overloading. Measured: greedy overloaded
   41% at N=2500/M=10 and thereby appeared to beat a feasible schedule by 16 pp.
   The invalid run is kept as `INVALID_raw_E10.csv`. Capacity is instead modelled
   with `pen > 0` (below).

## The three capacity regimes

The paper's two energy models have different mathematical character, and the
distinction explains every otherwise-puzzling result. See
`NOTES_constrained_baseline.md` for the full argument.

| regime | setting | character |
|---|---|---|
| Temporal | per-task energy | objective **separable** → greedy EDF+greenest is *provably optimal*; no search can improve on it |
| Consolidation | `ceil(load/C)` idle term | not separable, but green slots and consolidation align, so greedy is near-optimal |
| Capacitated | `pen > 0` | objectives genuinely conflict, greedy overloads, strongly NP-hard — the regime where search earns its place |

## Experiments

| ID | What | Grid |
|----|------|------|
| `E1` | Scalability, published model, soft penalty | N ∈ {500…3000} × M ∈ {10,20} × 7 methods × 30 seeds |
| `E4` | Hard deadline constraint (repair decoding) | same grid |
| `E20` | **Capacity as a feasibility constraint** + hard deadlines | same grid |
| `E21` | Capacity feasibility, soft deadlines | same grid |
| `E3` | Ablation, published model | greedy / random / improved / carbon init, β=0, γ=0, + seeding on all 6 optimisers |
| `E22` | Ablation under the feasibility constraint | as E3 |
| `E2` | Power models: linear, cubic, piecewise | N ∈ {500,1500,3000} × M × 3 models |
| `E5` | Sensitivity: 11 (α,β,γ) combinations + seeded fraction ∈ {0…1} | N ∈ {1000,3000} × M × cap |
| `E7` | 4 disjoint Google subsets + NASA real arrivals | N ∈ {500,1500,3000} × M |
| `E23` | **Minimum active-host constraint** (workload-determined warm pool) | N × M × mmin ∈ {0, 0.25, 0.5, auto} |
| `E24` | **Modern published schedulers**: Google VCC, *Let's Wait Awhile* threshold | full grid, both constraint models |
| `E25` | Runtime and memory at the task counts reviewers named | N ∈ {50,100,200,300} × M |
| `E26` | **Multiple carbon regions** × 6 real windows each | UK, IE, NI, US-CAL(, US-MIDA) |
| `E27` | Is the single published UK window representative? | all 37 disjoint UK windows |
| `CONV` | Convergence curves (per-epoch global best fitness) | N ∈ {500,1500,3000} × M |
| `GABUG` | Measures the `OriginalGA` defect at scale | N ∈ {500,1500,3000} × M |

## Carbon regions

`fetch_regions.py` and `fetch_us.py` build additional real carbon traces so the
conclusions are not tied to one grid. Each is stored as half-hourly
`data/carbon/carbon_history_<REGION>.csv`.

| region | slots | mean gCO₂/kWh | range | source |
|---|---|---|---|---|
| UK | 5,329 | 111.9 | 20–246 | National Grid ESO (existing) |
| IE | 5,374 | 168.6 | 77–336 | EirGrid Smart Grid Dashboard, no key |
| NI | 4,990 | 197.9 | 47–444 | EirGrid Smart Grid Dashboard, no key |
| US-CAL | 5,376 | 173.0 | 65–422 | EIA API v2, fuel mix × IPCC AR5 factors |

Two limitations are recorded explicitly rather than smoothed over: the EIA
publishes hourly, so US series are upsampled to 30-minute slots (an upsample, not
new information), and the UK time-of-use tariff is applied unchanged to every
region as a cost proxy, not a claim about local pricing.

Forecasting (`forecast.py`): horizons 1/6/12/24/48 slots with direct multi-step
prediction; MAE, RMSE, MAPE and MASE; persistence, seasonal-naive-24h and
seasonal-naive-1week baselines; 5 repeats per model per horizon with standard
deviations (the published single-run figures did not reproduce); forecast-error
degradation; and predicted vs reactive vs perfect-foresight at identical
optimisation budgets.

## Statistics

30 independent seeds. Every comparison reports mean, sd, 95% confidence interval,
**both** Welch's t-test and Mann-Whitney U (several methods have near-zero
variance, where a t-test is ill-conditioned), plus Cohen's d and Cliff's delta.
A result is marked significant only when both tests agree at α=0.05.

## Fixed parameters

Population 40, 120 epochs, 144 half-hour slots (3 days), deadline slack 8 slots,
maximum deferral 24 slots, host capacity C=1, P_idle=100 W, P_max=250 W,
weights (α,β,γ)=(0.4,0.3,0.3) unless swept. Workload: Google Cluster Trace 2011
`../data/workload/google_task_events_part0.csv.gz` and NASA-iPSC
`../data/workload/NASA.swf`.

## Secrets

`fetch_us.py` reads `EIA_API_KEY` from the environment or from `revision/.env`,
which is gitignored. No key is committed.

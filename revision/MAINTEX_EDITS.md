# main.tex revision checklist

Reviewer 2 §5 ("Minor but Important Presentation Corrections") plus the numerical and
structural changes the new results force. Line numbers refer to the submitted
`main.tex` (566 lines) in `carbon-aware-scheduler-updated/`.

---

## A. Reviewer 2 §5 — all eleven items

### 1. Grammar and wording
Specific instances found:

- **L60** "it enhances Whale Optimization with carbon-aware seeding and a fitness
  function that penalises deadline and capacity violations" — "enhances X with Y and a
  Z" is a strained parallel. Rewrite: "CA-WOA seeds part of the initial population with
  a carbon-aware schedule and minimises a weighted objective combining carbon, deadline
  violations and host overload."
- **L65** "up to about 87\%" — "up to about" is doubly vague. Give the figure and its
  configuration: "87.8 \% at 500 tasks on 10 hosts".
- **L87** "an effective and widely used optimiser" — unsupported editorialising about
  the method the paper is proposing. Delete.
- **L67** "CA-WOA scales safely" — "safely" is undefined. Replace with the measurable
  claim: "CA-WOA holds deadline violations at zero as load increases, whereas PSO, DE
  and GA violate 33-64 \% of deadlines".
- **L253** "It is emphasised that this is..." — passive throat-clearing; state the point
  directly.

### 2. Consistent terminology
Fix one term per concept throughout:

| Use | Not |
|---|---|
| carbon intensity (gCO2/kWh) | carbon factor, emission rate, CI value |
| carbon-aware scheduling | green scheduling, carbon scheduling |
| carbon reduction (%) | carbon saving, emission reduction, CO2 saving |
| emissions (kgCO2) | carbon, CO2 output |
| host | server, machine, VM (a host is not a VM -- keep these distinct) |
| slot | interval, timestep, period |

The host/VM conflation matters: the paper says "10 VMs and 20 VMs" in places and
"hosts" in others, but the model has *hosts* with capacity C and *tasks* placed on
them. There are no VMs in the simulator. Use "hosts" everywhere and say so once.

### 3. Define abbreviations at first use
Missing or late definitions: **SLA** (first used L65, never expanded), **FIFO** (L64),
**CI** (used for carbon intensity; also collides with confidence interval -- use "CI"
for carbon intensity only, and write "95 \% confidence interval" in full), **EDF**,
**MILP**, **NFE**, **VCC**, **MASE**, **SWF**. Add a nomenclature block after the
keywords, or expand on first use.

### 4. Figure readability, units, legends, error information
Every figure is regenerated, so fix at source:

- axis labels with units on all four figures: "Carbon reduction (\%)", "SLA violations
  (\%)", "Task count (N)", "Carbon intensity (gCO2/kWh)"
- **error bars on every mean** -- 95 \% confidence intervals over 30 seeds are now
  available in the `*_stats.csv` files; the submitted figures showed means with no
  dispersion at all
- legends inside the axes, not overlapping data
- minimum 9 pt tick labels (submitted figures are unreadable at print size)
- do not encode CA-WOA by colour alone -- add a marker or hatch, for greyscale printing
- state `n = 30 seeds` in each caption

### 5. Add standard deviations or confidence intervals to tables
`tab:capacity` and the scalability table currently report bare means. Every raw CSV has
one row per seed, so report `mean ± sd` and the 95 \% CI. Use the committed
`E20_stats.csv`, `E1_stats.csv` etc. rather than recomputing by hand.

### 6. Complete forecasting-model hyperparameters
`tab:forecast` omits most of what is needed to reproduce the models. Add: look-back
window (48 slots = 24 h), forecast horizons (1/6/12/24/48 slots), CNN filters/kernel
(64, 3, padding same, x2), pooling (MaxPool 2), LSTM units (48), GRU units (32),
gradient-boosting estimators and depth, optimiser (Adam), loss (MSE), epochs (40 CNN /
30 GRU), batch size (32), train/test split, feature set (scaled intensity plus sin/cos
of slot-of-day and day-of-week), scaling (min-max fitted on train only), and the seed.
State that figures are the mean of 5 repeats with standard deviations, because the
submitted single-run numbers did not reproduce.

### 7. Correct "Availability of data and materials: NA"
There is no such statement in `main.tex`, so it comes from the submission form or the
journal template. It must be replaced -- all three data sources are public:

> **Availability of data and materials.** All data are public. Workload traces: Google
> Cluster Trace 2011 (Reiss et al.) and the NASA-iPSC log from the Parallel Workloads
> Archive. Carbon intensity: UK National Grid ESO Carbon Intensity API, EirGrid Smart
> Grid Dashboard (Republic of Ireland and Northern Ireland), and the US EIA API v2
> (California and Mid-Atlantic; intensity derived from hourly fuel mix using IPCC AR5
> median lifecycle emission factors). All code, per-seed raw results and the validation
> suite are archived at [Zenodo DOI].

### 8. Persistent code repository link / DOI
See `ZENODO.md`. Add the DOI to the abstract footnote and to the availability
statement.

### 9. Clarify whether time-of-use tariff values are in the reported objective
**They are in the objective but they have no effect, and this must be stated.** The
tariff enters only through the cost term, and in the submitted configuration cost is
identical (0.755 GBP) for every method, so the term is inert. Say so explicitly:

> A time-of-use tariff (0.15 GBP/kWh off-peak, 0.30 GBP/kWh peak) is applied to compute
> monetary cost, which is reported for completeness. Because all methods schedule the
> same total workload and the tariff peak does not coincide with the carbon peak, cost
> is effectively constant across methods and does not influence the ranking. Cost is
> therefore reported but not optimised.

Also state that the UK tariff is applied unchanged to the non-UK regions as a cost
proxy, not a claim about local pricing.

### 10. Consistent units
Fix throughout: energy **kWh**, emissions **kgCO2** (totals) and **gCO2/kWh**
(intensity), power **W**, cost **GBP**, time **h** for makespan and **s** for runtime,
memory **MB**. The submitted text mixes gCO2 and kgCO2 without stating which. Give
every table column an explicit unit in its header.

### 11. Reference metadata, DOIs and formatting
See `NOTES_citations.md` §4 and §6: 12 verified additions, two metadata corrections, and
a note that HAPSO should not be cited until the version of record exists.

---

## B. Numerical and structural changes the new results require

### Claims that are now known to be wrong
| Location | Problem | Fix |
|---|---|---|
| Abstract L60-67 | "has not previously been used for carbon"; "up to about 87\%" | replacement abstract in `NOTES_novelty.md` §4 |
| §1 L87, L169 | novelty claim, hedged | delete; use the contribution list in `NOTES_novelty.md` §5 |
| §6.1 | "identical evaluation budgets" | false: HHO 8,908 vs 4,840. Report the measured NFE table |
| §6 headline | 83.25 \% attributed to CA-WOA | decompose: 80.51 pp consolidation, +2.06 timing, +0.68 CA-WOA over greedy |
| §6.4 forecasting | ensemble "12 \% better than LSTM" | does not reproduce (~7 \%); CNN-LSTM is the *worst* model at every horizon. Rewrite around the 12 h result: ensemble MAE 26.09 ± 0.48 vs seasonal-naive 37.03, MASE 0.59 |
| §5 config | "50 to 300 tasks", 3-5 seeds | 500-3000 tasks, 30 seeds, M ∈ {10,20} |

### Tables and figures to replace
`tab:config`, `tab:hyper`, `tab:capacity`, `tab:forecast`, `fig:temporal`,
`fig:capacityfig`, `fig:scale`, `fig:forecast` — all regenerate from the raw CSVs.

### New material to add
| Section | Content | Source |
|---|---|---|
| §4 (new subsection) | MILP formulation; separability lemma | `NOTES_constrained_baseline.md` |
| §4 (new subsection) | convergence analysis | `NOTES_convergence_analysis.md` |
| §5 | power models: linear / cubic / piecewise | `raw_E2.csv` |
| §5 | minimum active-host constraint | `raw_E23.csv` |
| §5 | heterogeneous fleets + startup overheads | `raw_E29.csv` |
| §5 | VCC and threshold baselines | `raw_E24.csv` |
| §6 | five carbon regions; 37 UK windows | `raw_E26/E27/E28.csv` |
| §6 | ablation, 5 arms | `raw_E3.csv`, `raw_E22.csv` |
| §6 | weight and seeded-fraction sensitivity | `raw_E5.csv` |
| §6 | runtime, memory, NFE, convergence curves | `raw_E25.csv`, `raw_CONV.csv` |
| §6 (new) | limitations | `NOTES_novelty.md` §3 |
| §7 | reframed conclusion | `NOTES_novelty.md` §5 |

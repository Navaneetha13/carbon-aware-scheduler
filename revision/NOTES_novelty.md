# Novelty claim: what to withdraw and what to claim instead

Reviewer 2: *"Strengthen the literature review and moderate unsupported claims such as
'first application.' Clearly identify the contribution of carbon-aware seeding,
SLA-aware optimization and predictive forecasting."*

---

## 1. What has to go

**Abstract, as submitted:**

> "This research investigates whether carbon-aware scheduling can be improved by
> applying the Whale Optimization Algorithm, **which has not previously been used for
> carbon**, to the problem..."

**Be precise about why.** Zhang & Wang (2024) apply an enhanced WOA to cloud task
scheduling but *not* to carbon, so they do not strictly refute the narrow claim. The
claim fails for three other reasons, and these are the ones to give if challenged:

1. It is **unfalsifiable in practice**. No search establishes that something has never
   been done, which is why reviewers treat "first application" as a red flag rather
   than a contribution.
2. The **gap it asserts is narrower than it appears**. Carbon-aware scheduling by
   metaheuristic is well established — evolutionary methods (Abbasi-khazaei & Rezvani
   2022), PSO and GWO variants, and carbon-aware VM placement (Khodayarseresht et al.
   2023). Only the specific pairing "WOA + carbon" is arguably untouched, and a
   contribution that rests on an algorithm-substitution gap is weak.
3. It is **already overtaken**. Carbon-aware cloud scheduling by multi-objective
   optimisation now has 2026 journal coverage (`danach2026carbon`,
   `ruparel2026carbon`), so a novelty claim framed this way ages badly between
   submission and publication.

Note also §1 line 169 hedges the same claim as "to the best of our knowledge it has not
been applied to carbon-aware scheduling". The hedge is honest but the claim still
carries no weight, and it displaces the contribution that *is* demonstrable. Delete
both, and claim C1-C6 below instead.

Two further claims need the same treatment:

- **§6 "83.25 % carbon reduction"** attributed to CA-WOA. The decomposition shows
  80.51 pp of that is consolidation, temporal shifting adds 2.06 pp, and CA-WOA adds
  0.68 pp over the greedy heuristic. The headline is real but almost entirely
  attributable to a mechanism that is not the contribution.
- **§6.1 equal evaluation budgets.** Measured: HHO receives 8,908 objective evaluations
  (min 8,698, max 9,136) against 4,840 for every other method. The claim is false as
  written.

## 2. What can be claimed, with the evidence

Each of the following is measured, reproducible from a committed CSV, and survives the
feasibility and constraint checks.

**C1 — Carbon-aware seeding is a transferable initialisation strategy, not a WOA
trick.** Seeding a fraction of the initial population with a carbon-aware greedy
schedule improves final solution quality in **59 of 60** (algorithm, configuration)
cells across two energy models and six optimisers — WOA, GWO, PSO, DE, HHO, GA. Mean
gain +2.51 pp under the published model (significant in 36/36) and +0.91 pp under the
capacity-feasibility model (significant in 21/24). It holds in all four carbon regions
(+0.64 to +1.98 pp, all p < 1e-9) and across heterogeneous fleets and startup
overheads (53/54 cells). *This is the paper's contribution, and it is stronger stated
this way than as a property of one algorithm.*

**C2 — The gain comes from the carbon information, not from better sampling.** An
"improved" initialisation using Latin-hypercube plus opposition-based sampling but no
carbon data is statistically indistinguishable from uniform random (+0.003 to +0.027
pp, p = 0.06–0.94). This isolates *what* the seeding contributes and is the cleanest
ablation result in the revision.

**C3 — Theory explains when search helps, and it is a proof, not a narrative.** In the
temporal model the objective is separable, so the greedy EDF + lowest-carbon-slot rule
is provably optimal and CA-WOA reaches the global optimum in one epoch — which is why
CA-WOA and greedy tie at exactly 11.17 % with sd 0.00. Once host capacity binds the
problem is a resource-constrained temporal bin-packing problem, greedy becomes
infeasible, and search is necessary. See `NOTES_constrained_baseline.md` and
`NOTES_convergence_analysis.md`.

**C4 — Under capacity limits, heuristics do not produce valid schedules.** Across five
carbon regions the greedy carbon-aware rule is feasible in only **25–42 %** of real
3-day windows, overloading hosts by up to 45.23 %. CA-WOA, plain WOA and Google's VCC
policy are feasible in 100 %. Greedy's higher raw carbon numbers are an infeasible
upper bound, not a competing result.

**C5 — CA-WOA beats a deployed production scheduler.** Against the virtual-capacity-
curve policy of Google's carbon-intelligent computing system, CA-WOA wins **12 of 12**
configurations (+0.35 to +4.48 pp) with both methods fully feasible, and +4.01 pp
overall across heterogeneous fleets. VCC's throttle floor was tuned over 8 values first
so the baseline is not straw-manned.

**C6 — Forecast-driven scheduling captures essentially all of the achievable benefit.**
A 12-hour-ahead ensemble forecast (MAE 25.35 gCO2/kWh) delivers **99–100 %** of the
saving available to perfect foresight at an identical optimisation budget.

## 3. What must be reported as a limitation

Stating these strengthens the paper; a reviewer will find them otherwise.

- **The seeding advantage nearly vanishes when a warm pool is enforced.** Under a
  workload-determined minimum active-host constraint, CA-WOA beats greedy in 12/12
  configurations (it loses 3/12 without the constraint) — but its edge over *plain*
  WOA falls to +0.04 pp, significant in only 1 of 12 cells. The metaheuristic earns its
  place in that regime; the seeding does not.
- **On a flat carbon grid the method does not help.** US Mid-Atlantic varies only 1.78x
  across the day against the UK's 12.30x; there CA-WOA loses by 0.13 pp and the seeding
  gain collapses to +0.14 pp. Temporal shifting needs a green trough to exploit. This
  identifies the deployment condition and is worth a paragraph.
- **CA-WOA loses to GA in 3 of 12 capacity-constrained cells**, all at M=10 (the
  tightest capacity), by 0.08–0.15 pp.
- **CA-WOA has higher variance than its rivals** — sd up to 0.97 against 0.11–0.21 for
  the others under some power models.
- **GA and GWO had not converged at the 120-epoch budget** (102 and 112 epochs to
  stabilise), so their results are lower bounds and the comparison understates them.
- **The submitted GA baseline performed no search** — `GA.OriginalGA` in mealpy 3.0.3
  evaluates only its initial population (40 evaluations, not 4,840). Results here use
  `GA.BaseGA`; the defect is measured in `raw_GABUG.csv`.

## 4. Replacement abstract

> Cloud data centres consume a growing share of global electricity, and because the
> carbon intensity of the grid varies through the day, *when* delay-tolerant workloads
> run strongly affects their emissions. This paper studies carbon-aware temporal
> shifting under realistic host-capacity constraints and asks a narrower question than
> is usual: not whether a metaheuristic can schedule for carbon, but *when* it is
> needed at all. We show analytically that without capacity limits the objective is
> separable, so a greedy earliest-deadline, lowest-carbon rule is provably optimal and
> no search can improve on it. Once host capacity binds, that rule becomes infeasible —
> across five real grid regions it violates capacity in 58-75 % of three-day windows,
> overloading hosts by up to 45 % — and the problem becomes NP-hard. In that regime we
> introduce carbon-aware seeding: initialising a fraction of a population-based
> optimiser with a carbon-aware greedy schedule. Evaluated on the Google Cluster Trace
> and the NASA-iPSC trace against carbon-intensity signals from the UK, Ireland,
> Northern Ireland and two US regions, with 30 independent seeds and task counts from
> 500 to 3000, seeding improves final solution quality in 59 of 60 algorithm-
> configuration cells across six optimisers, and provably cannot do worse than the
> heuristic it seeds. The resulting scheduler produces fully feasible, deadline-safe
> schedules and outperforms the virtual-capacity-curve policy of a deployed production
> carbon-aware scheduler in all twelve configurations tested. A 12-hour-ahead ensemble
> forecast captures 73-89 % of the advantage perfect foresight holds over reactive scheduling. We also
> report the boundary of the approach: on a grid whose carbon intensity varies only
> 1.78x across the day, and under an enforced minimum active-host constraint, the
> advantage of seeding disappears.

## 5. Contribution list for §1

1. A **regime analysis** separating the cases where carbon-aware scheduling is trivial
   (separable, greedy-optimal), near-trivial (consolidation-aligned) and genuinely hard
   (capacitated, NP-hard), with a proof for the first.
2. **Carbon-aware seeding**, shown to transfer across six population-based optimisers
   and to be provably bounded by the seed's quality.
3. An **isolating ablation** demonstrating that the benefit comes from carbon
   information rather than from more sophisticated sampling.
4. **Feasibility-first evaluation** across five real grid regions, showing that the
   standard greedy carbon-aware heuristic does not produce valid schedules under
   capacity limits.
5. A **reproducibility artefact**: 29 experiments, one row per configuration-method-
   seed, behind a 27-check validation gate that reproduces the published figures to
   machine precision before any new result is trusted.

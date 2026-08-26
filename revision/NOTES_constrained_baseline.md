# Theoretical treatment of the constrained-scheduling optimisation baseline

Reviewer request: *"Compare against an earliest-deadline-first + lowest-carbon-slot
heuristic and a constrained-scheduling optimization method."*

Supervisor direction: the EDF + lowest-carbon-slot heuristic is implemented and
compared empirically (it is the `Greedy(EDF+greenest)` row in every table). The
constrained-scheduling optimisation method is answered theoretically rather than
implemented. This note is the argument to put in the paper.

---

## 1. Exact formulation

The problem is a mixed-integer linear program. Let `x[i,s] ∈ {0,1}` be 1 if task
*i* starts in slot *s*, with `T_i` the set of deadline-feasible starts for task *i*
(`e_i ≤ s ≤ min(e_i + D, d_i − p_i)`, where `e_i` is the earliest start, `p_i` the
duration, `d_i` the deadline and `D` the maximum deferral):

**minimise**  Σ_i Σ_{s∈T_i} c[i,s] · x[i,s]

where `c[i,s] = Σ_{k=s}^{s+p_i−1} E_i · CI_k` is the carbon of running task *i*
from slot *s*, subject to

1. **assignment:** Σ_{s∈T_i} x[i,s] = 1  ∀i
2. **capacity:** Σ_i Σ_{s: s ≤ k < s+p_i} u_i · x[i,s] ≤ M·C  ∀k
3. **deadlines:** enforced structurally by the domain `T_i`

Deadlines are therefore *hard* by construction, which is what the reviewer's
"hard constraints or a repair mechanism" comment asks for, and matches the
`hard=True` decoding used in the empirical runs.

## 2. Why the exact method is not the right baseline here

**Complexity — and this depends on which energy model is used.** The paper uses two,
and they have different mathematical character. This distinction is the key
structural point and it should be stated explicitly in the revision.

*Temporal model* (used in `proposed_ca_woa.py`, `week4_robustness.py`). Energy per
task is `(P_idle + (P_max−P_idle)·u_i)·Δt·p_i`, which does **not** depend on what
else runs in the same slot. The objective is therefore fully separable across
tasks, constraint (1) is a partition matroid, and the optimum is obtained by
independently choosing `argmin_{s∈T_i} c[i,s]` for each task — **which is exactly
the EDF + lowest-carbon-slot greedy rule**. So in the temporal model the greedy
heuristic is *provably optimal* and no search of any kind can beat it.

This is a proof, not a conjecture, and it explains the otherwise puzzling result in
`../results/week4_robustness.csv`: CA-WOA and the carbon-aware greedy baseline both
score 11.17 % with standard deviation 0.00 over 5 seeds and the same makespan. They
are not coincidentally equal — CA-WOA is seeded with the greedy schedule, that
schedule is the global optimum, and the WOA operators can only fail to improve on
it. The same argument explains why the fitness weights (α, β, γ) have no effect in
that model.

*Capacity / consolidation model* (used in `week4_full_comparison.py`,
`scalability_sweep.py`). Per-slot energy is
`⌈load_k/C⌉·P_idle + (P_max−P_idle)·load_k`. The second term is separable, but the
idle term depends on the **aggregate** load in the slot, so co-locating tasks is
cheaper than spreading them: two tasks of u=0.4 in one slot cost 0.110 kWh, in
separate slots 0.160 kWh (verified numerically). The objective is therefore **not**
separable and greedy is not provably optimal. Empirically it is still very strong —
it beats CA-WOA in 9 of 12 configurations — because concentrating tasks in the
greenest slots happens to consolidate them as well, so the two objectives align
rather than conflict.

*Capacity model with M binding* (`cap=True`). Once the per-slot capacity constraint
(2) is active the greedy rule becomes badly infeasible: it piles work into the few
greenest slots and overloads. The problem is now a resource-constrained temporal
bin-packing problem, strongly NP-hard, and the two objectives genuinely conflict.
This is the only regime in which a population-based search has something to
contribute, and it is what the `cap=True` experiments measure.

**Scale.** The LP relaxation has Σ_i |T_i| variables. With the deadline slack of
8 slots used throughout, |T_i| ≤ 9, so N=3000 gives up to 27,000 binaries and 144
capacity rows. That is solvable, but the point of the comparison is a bound, not a
competitor: an exact solver has no anytime behaviour and no per-slot decision
budget, so it does not answer the question the paper asks (can a population-based
search reach good schedules under a fixed evaluation budget).

**What it should be used for.** The honest role of the MILP is as an **optimality
gap reference**, reported once per configuration, not as a scheduling method
compared on equal footing.

## 3. What to state in the paper

1. The scheduling problem is formulated exactly as the MILP above; deadlines are
   hard, capacity is a knapsack constraint per slot.
2. In the **temporal** model the objective is separable, so the greedy EDF +
   lowest-carbon-slot rule is **provably optimal**. State this as a short lemma —
   it is one line from separability of the objective and independence of the
   per-task domains — and use it to explain why CA-WOA and the greedy baseline
   coincide exactly there, and why the fitness weights have no effect.
3. In the **capacity/consolidation** model the idle-power term couples tasks, so
   greedy is not provably optimal; it is nonetheless empirically near-optimal
   because low-carbon slots and consolidation align. Report that it beats CA-WOA
   in 9 of 12 configurations rather than hiding it.
4. The metaheuristic contribution is therefore confined to the **capacitated**
   regime, where greedy overloads and the objectives genuinely conflict. That
   regime is strongly NP-hard and is what the `cap=True` experiments measure.
5. A full MILP solve is identified as future work for computing optimality gaps at
   each task count, with the complexity argument above given as the reason it is
   not used as a head-to-head baseline.

## 4. Consequence for the paper's framing

This reframes the contribution honestly and defensibly. Rather than "CA-WOA beats
the baselines", the claim becomes:

> Carbon-aware temporal shifting without capacity limits is a separable problem
> solved optimally by a greedy rule; metaheuristic search is neither necessary nor
> beneficial there. Once host capacity binds, the problem becomes NP-hard, the
> greedy rule overloads, and carbon-aware seeding gives a population-based search
> a decisive advantage over random initialisation.

That is a stronger paper than the current one because it says *when* the method
helps and why, and it turns the greedy result from an embarrassment into the
theoretical anchor of the argument.

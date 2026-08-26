# Theoretical convergence analysis of CA-WOA

Reviewer 3, comment 8: *"Add theoretical convergence analysis."*

This note gives the analysis to insert as a new subsection in the design chapter. It
is deliberately honest about what can and cannot be proved: the standard result for
this class of algorithm is convergence of the *best-so-far* sequence, **not**
convergence to the global optimum, and the distinction turns out to explain the
paper's empirical results rather than merely satisfying a reviewer.

---

## 1. The search space is finite

Each task *i* is assigned an integer start slot in

    S_i = { e_i , ... , e_i + r_i },      r_i = min(MAX_DEFER, H - p_i - e_i)

and, under hard decoding, additionally `s_i ≤ d_i - p_i`. The decoded search space is
therefore the finite product `S = S_1 x ... x S_N`, with

    |S| = prod_i (r_i + 1)  ≤  (MAX_DEFER + 1)^N = 25^N .

The continuous vector `x ∈ [0,1]^N` that WOA manipulates is only a parameterisation:
`decode(x) = e + round(x * r)` is a many-to-one map onto `S`. Two consequences follow
immediately, and both are used below:

1. The objective takes **finitely many values**, so there is a non-zero gap
   `δ = min{ F(u) - F(v) : u,v ∈ S, F(u) ≠ F(v) } > 0` between distinct fitness levels.
2. Convergence questions are questions about a **finite** state space, so no
   measure-theoretic machinery is needed.

## 2. The population sequence is a non-homogeneous Markov chain

Let `P_t = (x_t^1, ..., x_t^n)` be the population at epoch *t* with `n = 40`. WOA
updates each agent using only `P_t`, the incumbent best `x*_t`, and fresh randomness,
so

    Pr( P_{t+1} = q | P_t = p, P_{t-1}, ..., P_0 ) = Pr( P_{t+1} = q | P_t = p ) ,

i.e. `{P_t}` is a Markov chain. It is **not** homogeneous. WOA's control parameter

    a(t) = 2 - 2t/T

decreases linearly to zero, and the branch taken depends on `A = 2a·r - a`:

- `|A| < 1` — *encircling / spiral*: move toward `x*_t` (exploitation);
- `|A| ≥ 1` — *random agent*: move toward a randomly chosen population member
  (exploration).

Since `|A| ≤ a(t)`, the exploration branch becomes **unreachable** once `a(t) < 1`,
that is for `t > T/2`. The transition kernel therefore depends on *t*, and the chain
is non-homogeneous. This single fact drives everything that follows.

## 3. Convergence of the best-so-far sequence

**Lemma 1 (monotonicity).** mealpy's `OriginalWOA` retains the incumbent best across
epochs, so with `F*_t = F(x*_t)` the sequence `F*_0 ≥ F*_1 ≥ F*_2 ≥ ...` is
non-increasing, and bounded below by `F_min = min_{u ∈ S} F(u) ≥ 0`.

**Theorem 1 (almost-sure convergence).** `F*_t` converges almost surely to a limit
`F*_∞ ≥ F_min`, and it does so in at most `|F(S)|` strict decreases.

*Proof.* A non-increasing sequence bounded below converges. By §1 the objective takes
finitely many values separated by at least `δ`, so at most `(F*_0 - F_min)/δ` strict
decreases can occur; after the last one the sequence is constant. ∎

This is the honest form of the convergence guarantee, and it is all that elitism buys.
It says nothing about *which* value `F*_∞` takes.

## 4. Global convergence does not hold — and that is the useful part

The standard sufficient condition for a stochastic optimiser to converge to the global
optimum in probability is that the one-step transition probability into any
neighbourhood of the optimum stays bounded away from zero:

    exists ε > 0 :  Pr( x*_{t+1} ∈ B | P_t ) ≥ ε   for all t and all P_t,     (*)

whence `Pr(F*_t > F_min) ≤ (1-ε)^t → 0`. For WOA, (*) **fails**. Once `t > T/2` the
exploration branch is closed and every agent moves toward `x*_t`; the reachable set
contracts to a neighbourhood of the incumbent. Formally, the set

    R_t = { u ∈ S : Pr(x*_{t+1} = u | P_t) > 0 }

is non-increasing in *t* for `t > T/2`, so if the global optimiser leaves `R_t` at some
epoch it can never re-enter. WOA is therefore **not** a global-convergence algorithm in
the sense of (*); it converges almost surely to a *local* optimum determined jointly by
the initial population and the random seed.

Two things follow, and they are exactly what the experiments measure.

**Corollary 1 (initialisation is not a heuristic detail).** For a chain satisfying (*)
the limit is independent of the initial distribution. Because WOA violates (*), the
distribution of `F*_∞` depends on the distribution of `P_0`. Changing the
initialisation therefore changes the limit *in principle*, not merely the convergence
rate — which is why carbon-aware seeding produces a statistically significant
improvement in 59 of 60 (algorithm, configuration) cells rather than merely reaching
the same answer sooner. It also predicts the observed variance reduction: seeding
concentrates `P_0` on a region of `S` with better local optima, shrinking the spread of
`F*_∞` (measured: sd 0.381 → 0.111 at N=1000).

**Corollary 2 (the seed is a bound).** If `P_0` contains a solution `u`, then
`F*_∞ ≤ F(u)` almost surely, by Lemma 1. CA-WOA seeds the greedy EDF+lowest-carbon
schedule, so CA-WOA can never be worse than that heuristic on the fitness it
optimises. This is a proof, and it is worth stating: it converts "CA-WOA is at least as
good as greedy" from an empirical observation into a guarantee.

## 5. When the local optimum is the global one

The separability argument in `NOTES_constrained_baseline.md` closes the gap in one
regime. In the **temporal** model the objective is separable across tasks and the
per-task domains are independent, so the greedy rule `argmin_{s ∈ S_i} c[i,s]` attains
the global optimum. Combining that with Corollary 2:

**Proposition 1.** In the temporal model, CA-WOA converges almost surely to the global
optimum, in one epoch.

*Proof.* The greedy schedule `u_g` is globally optimal by separability, and it is in
`P_0` by construction, so `F*_0 = F(u_g) = F_min`; Lemma 1 gives `F*_t = F_min` for all
*t*. ∎

This is a genuine (if narrow) global-convergence result, and it explains the otherwise
puzzling observation that CA-WOA and the greedy baseline score identically — 11.17 %
with standard deviation 0.00 over five seeds in `../results/week4_robustness.csv`. They
are not coincidentally equal; they are provably equal, and no amount of extra search
budget can separate them.

In the capacitated regime the objective is not separable, `u_g` is generally
infeasible, and Proposition 1 does not apply — which is precisely where the search has
something to contribute and where the measured gains appear.

## 6. Budget, not asymptotics, is the binding constraint

Theorem 1 is asymptotic; the experiments use a fixed budget of `T = 120` epochs with
`n = 40`, i.e. 4,840 objective evaluations. The convergence curves in `raw_CONV.csv`
show where each method actually sits relative to its own limit at that budget:

| method | epochs to within 1 % of its final value |
|---|---|
| CA-WOA | 8 |
| HHO | 11 |
| WOA | 16 |
| DE | 51 |
| GA | 102 |
| GWO | 112 |

CA-WOA has effectively converged by epoch 8, so the reported figures are its limit and
a larger budget would not change them. GA and GWO have **not** converged by epoch 120,
so their results are lower bounds on their achievable performance and the comparison
understates them. That caveat should be stated in the evaluation chapter rather than
left for a reader to infer.

## 7. What to put in the paper

1. State the finite search space and the resulting fitness gap `δ` (§1).
2. State that `{P_t}` is a non-homogeneous Markov chain because `a(t) → 0` (§2).
3. Give Lemma 1 and Theorem 1: elitism plus finiteness gives almost-sure convergence
   of the best-so-far sequence in finitely many improvements.
4. State plainly that condition (*) fails, so WOA has **no** global-convergence
   guarantee, and that this is a property of WOA rather than of the carbon-aware
   extension.
5. Give Corollaries 1 and 2 — initialisation changes the limit, and the seed bounds it.
   These are the theoretical justification for carbon-aware seeding, and they are what
   the ablation measures.
6. Give Proposition 1 for the temporal model, and use it to explain the exact tie with
   the greedy baseline.
7. Report the convergence table in §6 and the GA/GWO budget caveat.

## 8. Sources for the framework

The Markov-chain argument used here follows the standard treatment of swarm
metaheuristics: convergence analyses of the Bat Algorithm and Flower Pollination
Algorithm establish almost-sure convergence of the elite sequence under exactly this
finite-state, non-homogeneous-chain formulation, and identify the vanishing-exploration
condition as the obstacle to global convergence. Cite one such analysis alongside the
original WOA paper (`mirjalili2016whale`) rather than claiming the framework as new.

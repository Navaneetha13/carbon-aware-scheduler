# Energy-Efficient & Carbon-Aware Cloud Scheduling — Project Plan & 10-Week Timetable
**Navaneetha Thalakokkula (23213914) · MSc Cloud Computing · National College of Ireland**

Status: Weeks 1–2 done (10-paper literature review, comparison table, factor-coverage matrix,
preliminary Mealpy simulation on Google Cluster Trace). This document plans Weeks 3–12.

---

## 1. Platform & Tools (answer for the professor: "what platform?")

| Layer | Tool (verified, real) | Role |
|---|---|---|
| Language | **Python 3.10+** | All implementation |
| Dev / demo environment | **Jupyter Notebook** (in VS Code) | Lets us show *formula → code → result* in one screen — ideal for the formula screenshots |
| Metaheuristics | **Mealpy 3.0.3** (Nguyen Van Thieu; MIT) | GWO, PSO, DE, WOA, HHO, GA, ACO — the scheduling optimizers |
| Workload data | **Google Cluster Trace** (real, public; using a workable subset) | Tasks: CPU/mem demand, arrival, duration |
| Carbon data | **UK Carbon Intensity API** (National Grid ESO — free, no key) + ElectricityMaps CSVs | Real gCO₂/kWh, 30-min + 48h forecast |
| Renewables | **Open-Meteo API** (free, no key) | Real solar/wind for battery/solar model |
| ML | **TensorFlow / Keras** | LSTM forecasting |
| Clustering / utils | **scikit-learn, numpy, pandas** | k-means task clustering, data handling |
| Plots | **matplotlib** | Result figures |

> Everything is real, free, and original code. Libraries and datasets are **used and cited** —
> not copied. No fabricated data or results at any step (see §4 safeguards).

### 1A. Data Sources & APIs — honest usage decision (of the 10 proposed)
We do **not** need all 10 (that would be scope bloat). Verdict per source:

| # | Source | Real / Access | Decision |
|---|--------|---------------|----------|
| 1 | **Google Cluster Trace** | Real, public (full is huge; use subset) | ✅ **USE — primary workload** (keeps Week-1 consistency) |
| 2 | Alibaba Cluster Trace | Real, public (~48 GB; `batch_task` 125 MB) | ➕ Optional — 2nd benchmark only if time allows |
| 3 | Azure Public Dataset | Real, public (~78 GB) | ⏭️ Skip — redundant with Google, very large |
| 4 | **ElectricityMap** | Real; **live API now PAID**, free historical CSVs (login) | ✅ **USE — free CSVs** for multi-region carbon |
| 5 | WattTime API | Real; **free tier = 1 US region only** | ➕ Optional/limited — not primary |
| 6 | **National Grid ESO Carbon Intensity API** | Real, **FREE, no key**; 30-min + forecast + regional | ✅ **USE — PRIMARY carbon source** |
| 7 | **SPECpower** | Real SPEC benchmark (power figures) | ✅ **USE — source for P_idle/P_max** in the power formula |
| 8 | Cloud Carbon Footprint (CCF) | Real open-source tool/coefficients | ➕ Optional — cloud emission coefficients / methodology citation |
| 9 | "Carbon Trace (MIT)" | ⚠️ **Unverified** — no matching MIT dataset found; closest is *Climate TRACE* (a nonprofit, different purpose) | ❓ **Verify exact source or DROP** — not needed |
| 10 | **OpenWeather API** | Real, free tier (needs free key) | ✅ **USE — solar/wind for battery model** (or Open-Meteo, no key) |

**Essential real set we proceed with:** Google trace (subset) · National Grid ESO API · ElectricityMap CSVs · SPECpower figures · OpenWeather. Alibaba + CCF optional. WattTime/Azure/Carbon-Trace set aside with reasons above.

---

## 2. Core Formulas (answer for the professor: "screenshots with formulas")

These are implemented in code and shown in the Jupyter notebook (markdown LaTeX next to the code
that computes them, next to the real output). Standard, established formulas:

**(1) Host CPU utilization**  `u_h(t) = Σ(CPU demand of VMs on host h) / capacity_h`,  with `0 ≤ u ≤ 1`

**(2) Server power model (linear, SPECpower / Beloglazov)**
`P_h(u) = P_idle + (P_max − P_idle) · u`,  with `P_idle ≈ 0.65 · P_max` (Barroso & Hölzle, 2007)

**(3) Energy**  `E = Σ_t P_h(u_h(t)) · Δt`  (Wh → /1000 → kWh)

**(4) Carbon emissions**  `CO₂ = Σ_t E_t · CI(t, region)`  where `CI` = grid carbon intensity (gCO₂/kWh); /1000 → kg

**(5) Electricity cost**  `Cost = Σ_t E_t · Price(t)`  (Price = time-of-use $/kWh)

**(6) Makespan**  `makespan = max_i (finish_time_i)`

**(7) SLA-violation rate**  `SLAV = |{ i : finish_i > deadline_i }| / N × 100%`

**(8) Average resource utilization**  `U = Σ_h Σ_t u_h(t) / (active_hosts · T)`

**(9) Min–max normalization**  `x̂ = (x − x_min) / (x_max − x_min)`

**(10) Multi-objective fitness (what Mealpy minimizes)**
`minimize  F(s) = w₁·Ê(s) + w₂·CÔ₂(s) + w₃·Cost̂(s)`,  with `w₁+w₂+w₃ = 1`
`subject to  SLAV(s) ≤ θ_SLA  and  Cost(s) ≤ Budget`
where `s` = the schedule (task→VM/host assignment + timing). Constraint handling via penalty:
`F'(s) = F(s) + λ · max(0, SLAV(s) − θ_SLA)`

**(11) Improvement vs baseline**  `CarbonReduction% = (CO₂_baseline − CO₂_proposed) / CO₂_baseline × 100`

---

## 3. The 10-Week Timetable (Weeks 3–12)

Each week ends with a **screenshot/demo-able artifact** so live progress is always showable.

| Wk | Focus | Concrete output to show the professor |
|----|-------|----------------------------------------|
| **3 (now)** | **Foundation + formulas + 1 algorithm running** | Jupyter notebook: real data loaded, formulas (1)–(11) shown, **GWO** run via Mealpy producing real energy/carbon/cost/SLA numbers + 1 plot. **← this week's live demo** |
| **4** | Baselines + all metaheuristics on one fair benchmark | FIFO/Round-Robin baseline + GWO/PSO/DE/WOA/HHO/GA run on identical trace & metrics → comparison table (extends Week-1 sim, now reproducible) |
| **5** | **Pillar 1 — live carbon data** | UK Carbon Intensity API integrated (live + forecast); "follow-the-renewables" + temporal job deferral working on real grid data |
| **6** | **Pillar 2 — ML forecasting (LSTM)** | LSTM predicts carbon intensity + workload; forecast-driven scheduling beats reactive; show MAE/RMSE + forecast-vs-actual plot |
| **7** | **Pillar 3 — battery/solar hybrid** | Grid + solar + battery decision logic with real Open-Meteo data; energy/carbon recomputed with storage |
| **8** | Integrate all 3 pillars → **proposed hybrid scheduler** | Full proposed algorithm: metaheuristic optimizing energy+carbon+cost s.t. SLA, using live carbon + LSTM + battery — running end-to-end |
| **9** | **Full evaluation** vs all baselines | Proposed vs FIFO / energy-aware / carbon-aware / metaheuristic baselines on the full trace; sensitivity analysis (weights, deadline slack) |
| **10** | Results consolidation + reproducibility | All final tables/figures; fixed seeds; statistical check; README so results reproduce from one command |
| **11** | **Thesis writing** | Implementation + Evaluation chapters drafted; results integrated; Future Work written |
| **12** | Finalize + submit | Proofread, config/setup manual, final live demo, submission package |

**Buffer logic:** the headline results exist by Week 9; Weeks 10–12 are consolidation + writing,
so any slippage in the harder weeks (6 LSTM, 7 battery) is absorbed without risking the deadline.

---

## 4. Academic-integrity & "no-mistakes" safeguards (proceed carefully)

- **No fabricated data or results — ever.** Every number shown comes from a real run on real data. If something isn't done yet, we label it "in progress," never invent a result.
- **Original code only.** Datasets/libraries used and cited; nothing copied from another repo or pasted into the report.
- **Reproducibility:** fixed random seeds + documented run commands → the professor can re-run and get the same numbers.
- **Verification each week:** sanity checks (e.g., a carbon-aware schedule must never emit *more* CO₂ than the baseline on the same data) catch bugs before they reach a submission.
- **Citations verified:** all 12 Week-1 papers confirmed real via Crossref (see `literature/00_INDEX.md`); a few author/year fixes noted there.
- **Stay aligned with the approved proposal** so the methodology matches what was signed off.

---

## 5. This week (Week 3) — concrete plan for the live demo
Build one Jupyter notebook that:
1. Loads a Google Cluster Trace subset (real) + real UK carbon-intensity data.
2. Displays formulas (1)–(11) in LaTeX markdown cells (the screenshots the professor wants).
3. Defines the scheduling problem + fitness function (10) for **Mealpy**.
4. Runs **GWO** end-to-end → prints real Energy (kWh), Carbon (kg), Cost, SLA% + one comparison plot.
5. Clear "platform used" summary cell (§1) for the professor.

This is the smallest real slice that demonstrates progress and contains formulas, the platform,
and live output — directly satisfying all three of the professor's asks.

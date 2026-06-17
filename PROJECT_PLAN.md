# Project Master Plan — Energy-Efficient & Carbon-Aware Cloud Scheduling
**Navaneetha Thalakokkula (23213914) · MSc Cloud Computing · National College of Ireland**
**Duration: 12 weeks · This file = the full roadmap (what's done, current, and upcoming).**

> Companion files: `PROJECT_TIMETABLE.md` (formulas, platform, data-source decisions) ·
> `literature/00_INDEX.md` (the 10 papers, verified) · `notebooks/` (the simulation).

---

## Status snapshot (as of Week 3)
- ✅ **Done:** literature review + the simulation of existing algorithms, on real data, in a notebook.
- 🔄 **Now:** showing the simulation + formulas + platform at the 3rd supervisor connect.
- ⬜ **Next:** add the 3 novelty pillars (multi-region live carbon → LSTM forecasting → battery), then full evaluation and write-up.

**The thesis novelty (the goal of Weeks 5–8):** combine three things no single paper does together —
① **live grid carbon data**, ② **ML (LSTM) forecasting** of workload + renewables, ③ **battery/solar storage** —
on top of the best metaheuristic scheduler.

---

## 12-Week Roadmap

| Wk | Focus | Deliverable | Status |
|----|-------|-------------|--------|
| **1–2** | Literature review: 10 recent papers, comparison table, factor-coverage matrix; first Mealpy simulation | Document 1 + Document 2 | ✅ Done |
| **3** | Platform + formulas + **simulating existing algorithms** (GWO/PSO/DE/WOA/HHO/GA on a common benchmark, real data) | Colab + local notebooks, comparison table | ✅ Done |
| **4** | Consolidate the algorithm comparison: add **makespan + utilisation** metrics, **multi-seed** robustness, finalise baselines; align workload (Google trace if required) | Reproducible comparison + stats | ⬜ Upcoming |
| **5** | **Pillar ① — multi-region live carbon** ("follow the renewables": shift jobs across regions by carbon intensity) | Geo-aware scheduling on real multi-region data | ⬜ |
| **6** | **Pillar ② — LSTM forecasting** of carbon intensity + workload; feed forecasts into the scheduler | LSTM model + forecast-driven scheduling | ⬜ |
| **7** | **Pillar ③ — battery/solar hybrid** energy management (grid + solar + battery; OpenWeather data) | Hybrid energy model integrated | ⬜ |
| **8** | **Integrate all 3 pillars** into the proposed hybrid scheduler (best metaheuristic + live carbon + LSTM + battery) | The complete proposed system | ⬜ |
| **9** | **Full evaluation**: proposed vs all baselines (FIFO, energy-aware, carbon-aware, metaheuristics); full factor matrix; sensitivity analysis | Final results tables + figures | ⬜ |
| **10** | Results consolidation, **statistical validation**, reproducibility (seeds, README), polish all charts | Finalised, reproducible artifact | ⬜ |
| **11** | **Thesis writing**: Implementation + Evaluation chapters; integrate results; Future Work | Draft dissertation chapters | ⬜ |
| **12** | Finalise: proofread, config/setup manual, final demo, submission package | Final submission | ⬜ |

**Buffer logic:** headline results exist by Week 9; Weeks 10–12 are consolidation + writing, so any
slippage in the harder pillar weeks (6 LSTM, 7 battery) is absorbed without risking the deadline.

---

## What each upcoming week involves (detail)

### Week 4 — Strengthen the existing-algorithm comparison
- Add **makespan**, **resource utilisation** to the metrics (extend the factor matrix coverage).
- Run each algorithm with **multiple random seeds** and report mean ± std (statistical fairness).
- Optionally add **ACO/SMA** to widen the algorithm spread.
- **Decision point:** if the supervisor requires the **Google Cluster Trace**, swap it in (same code).

### Week 5 — Pillar ① Multi-region live carbon
- Add 2–3 regions with different carbon profiles (UK National Grid + ElectricityMaps CSVs).
- Implement **spatial "follow-the-renewables"** placement + temporal shifting, subject to latency.

### Week 6 — Pillar ② LSTM forecasting
- Train an **LSTM** to predict near-future carbon intensity (and workload peaks).
- Scheduler defers jobs toward **forecasted** low-carbon windows (not just current values).
- Report forecast accuracy (MAE/RMSE) + that forecast-driven beats reactive.

### Week 7 — Pillar ③ Battery / solar hybrid
- Real solar/wind via **OpenWeather/Open-Meteo**; model **grid + solar + battery** decision logic.
- Use solar → battery → grid; recompute energy/carbon with storage.

### Week 8 — Integrated proposed scheduler
- Combine best metaheuristic + live carbon + LSTM + battery into one carbon-aware scheduler.

### Week 9 — Full evaluation
- Compare proposed vs the three baselines (non-carbon-aware, energy-aware, carbon-aware) + metaheuristics.
- Target outcomes from the proposal: carbon ↓20–35%, energy ↓15–25%, SLA <5%, cost ↓10–15%.

---

## Open decisions to confirm with supervisor
1. **Workload:** ✅ **Resolved — now using the real Google Cluster Trace 2011** (downloaded from Google's public bucket over HTTPS; no AWS, no login). Matches what was presented to the supervisor. (NASA-iPSC remains available as a second benchmark.)
2. **Simulation tool:** proposal named **CloudSim Plus** (Java); we use **Python + Mealpy** (Week-1 direction). Confirm Python/Mealpy is acceptable.
3. **Power values & tariff:** currently documented assumptions (SPECpower-style 100/250 W; ToU £0.30/£0.15). Use real SPECpower figures if he wants.
4. **Section 5.5 (federated learning, blockchain, hardware co-design):** written as Future Work, not implemented (standard).

---

## Integrity guardrails (apply every week)
- **Real data only** (workload + carbon are real; power/price are *labelled* assumptions). No fabricated numbers.
- **Original code**, datasets/libraries cited (Mealpy: Van Thieu & Mirjalili 2023, DOI 10.1016/j.sysarc.2023.102871).
- **Reproducible:** fixed seeds, documented run commands.
- Report results honestly even when an algorithm underperforms (e.g., GA).

---

## Current results (Week 3 — for reference)
Simulating existing algorithms on 60 real **Google Cluster Trace** tasks + real UK carbon (level playing field):

| Algorithm | Carbon ↓ | SLA viol. |
|---|---|---|
| **HHO** (best all-rounder) | **11.1%** | 0% |
| DE | 11.5% | 18% |
| GA | 11.3% | 48% |
| PSO | 9.1% | 13% |
| GWO | 2.8% | 0% |
| WOA | 1.2% | 0% |

(Baseline = FIFO/Round-Robin. Best *all-rounder* = highest carbon cut with 0 SLA violations → HHO here.)

---

## File map
```
carbon-aware-scheduler/
├── PROJECT_PLAN.md            ← this file (master roadmap)
├── PROJECT_TIMETABLE.md       ← formulas, platform, data-source decisions
├── notebooks/
│   ├── carbon_aware_scheduling_COLAB.ipynb   ← simulation for Google Colab
│   ├── carbon_aware_scheduling.ipynb         ← simulation for VS Code/Jupyter
│   └── *.html                                ← rendered copies
├── src/                       ← all source code
├── data/                      ← real NASA workload + real carbon data
├── results/                   ← comparison CSV + charts
└── literature/                ← 5 downloaded papers + verified 00_INDEX.md
```

# Carbon-Aware & Energy-Efficient Cloud Task Scheduling

MSc Cloud Computing dissertation — **Navaneetha Thalakokkula**, National College of Ireland.

Simulates existing metaheuristic scheduling algorithms (**GWO, PSO, DE, WOA, HHO, GA**) on a common
benchmark — the real **Google Cluster Trace 2011** and real **UK National Grid** carbon-intensity data —
comparing **energy, carbon, cost and SLA** on a level playing field. Deferrable jobs are shifted into
cleaner-energy periods. Built with **Python + [Mealpy](https://github.com/thieu1995/mealpy)**.

## Run it
- **Google Colab:** open `notebooks/carbon_aware_scheduling_COLAB.ipynb` → *Run all*.
  It installs Mealpy and downloads the real Google trace automatically — no setup, no AWS.
- **Locally:** `notebooks/carbon_aware_scheduling.ipynb` (Python 3.10 + the listed libraries).

## Structure
| Path | Contents |
|------|----------|
| `notebooks/` | The simulation (Colab + local), with rendered HTML copies |
| `src/` | Source code (simulation, comparison, notebook builders) |
| `data/` | Real workload + carbon data |
| `results/` | Comparison tables and charts |
| `literature/` | Reviewed papers + verified index |
| `PROJECT_PLAN.md`, `PROJECT_TIMETABLE.md` | Roadmap, formulas, platform & data decisions |

## Platform
Python 3.10 · Mealpy 3.0.3 (Van Thieu & Mirjalili, 2023) · NumPy · pandas · scikit-learn · Matplotlib.

## Method (current stage)
Each metaheuristic minimises a carbon objective with an SLA-deadline penalty, on the same workload and
the same real carbon-intensity series; results are compared against a non-carbon-aware FIFO/Round-Robin
baseline. Next stages: multi-region carbon, LSTM forecasting, and battery/solar storage.

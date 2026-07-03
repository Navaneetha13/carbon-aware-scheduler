# Datasets & References
Data sources used in this project, where to download them, and the papers to cite.

## Datasets

### 1. Google Cluster Trace 2011 — cloud workload (jobs, CPU/mem requests)
- **Where to get (free, public):** https://github.com/google/cluster-data  (data hosted on Google's public bucket)
- **Cite (schema report):** Reiss, C., Wilkes, J., Hellerstein, J. L. *Google cluster-usage traces: format + schema.* Google Inc. Technical Report, 2011 (rev. 2014).
- **Cite (analysis paper):** Reiss, C., Tumanov, A., Ganger, G. R., Katz, R. H., Kozuch, M. A. *Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis.* ACM Symposium on Cloud Computing (SoCC), 2012.  *(verified on Crossref/ACM DL)*

### 2. UK Grid Carbon Intensity — real gCO₂/kWh (carbon data + 111-day history)
- **Where to get (free API, no key):** https://carbonintensity.org.uk
- **Source / how to cite:** National Grid ESO, in partnership with the University of Oxford (Environmental Change Institute). Methodology documented on the site. *(No single journal paper — cite the Carbon Intensity API and its methodology page.)*

### 3. NASA-iPSC — job-arrival trace (used for workload forecasting)
- **Where to get (free):** https://www.cs.huji.ac.il/labs/parallel/workload/  (Parallel Workloads Archive)
- **Cite:** Feitelson, D. G., Tsafrir, D., Krakov, D. *Experience with using the Parallel Workloads Archive.* Journal of Parallel and Distributed Computing, 74(10):2967–2982, 2014. **DOI: 10.1016/j.jpdc.2014.06.013**  *(verified)*

### 4. Open-Meteo — solar irradiance (for the battery/solar pillar, next week)
- **Where to get (free API, no key):** https://open-meteo.com  (based on ECMWF ERA5 reanalysis)
- **Cite (ERA5):** Hersbach, H., et al. *The ERA5 global reanalysis.* Quarterly Journal of the Royal Meteorological Society, 146:1999–2049, 2020. **DOI: 10.1002/qj.3803**

## Models / tools
- **Mealpy** (metaheuristics library — WOA/GWO/PSO/DE/HHO/GA): Van Thieu, N., Mirjalili, S. *MEALPY: An open-source library for latest meta-heuristic algorithms in Python.* Journal of Systems Architecture, 2023. **DOI: 10.1016/j.sysarc.2023.102871**  *(verified)*
- **LSTM:** Hochreiter, S., Schmidhuber, J. *Long Short-Term Memory.* Neural Computation, 9(8):1735–1780, 1997. **DOI: 10.1162/neco.1997.9.8.1735**  *(verified)*
- **GRU:** Cho, K., et al. *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* EMNLP 2014. **DOI: 10.3115/v1/D14-1179**  *(verified)*

*(The 10 reviewed scheduling-algorithm papers, all verified, are in `literature/00_INDEX.md`.)*

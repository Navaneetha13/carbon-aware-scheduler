# Citation review and literature-review strengthening

Addresses Reviewer 3 comment 11 ("Review all citations. The introduction fails to
engage with recent, highly relevant works from 2023–2025") and Reviewer 2's novelty
comment ("Strengthen the literature review and moderate unsupported claims").

**Every DOI below was resolved against Crossref before being written here.** Nothing
in this file is transcribed from a search-result snippet.

---

## 1. State of the submitted bibliography

`refs.bib` holds 19 entries. The newest is 2023 (`radovanovic2023carbon`,
`vanthieu2023mealpy`). There is **nothing from 2024–2026**, which is what the reviewer
noticed.

Separately: `../literature/00_INDEX.md` documents **12 papers that were verified in
Week 1 and then never cited**. Four of them are described in that very index as "the
explicitly carbon-aware works that justify the thesis gap". Citing the gap-defining
literature is not optional, and its absence is the single most damaging omission in the
submitted reference list.

## 2. The "first application" claim must be withdrawn

The abstract states the paper applies WOA "which has not previously been used for
carbon"; §1 (line 169) hedges the same point as "to the best of our knowledge". The
claim should be withdrawn -- see `NOTES_novelty.md` §1 for the precise reasoning, which
matters if a reviewer challenges it. In short: it is unfalsifiable, the gap it asserts
is an algorithm substitution rather than a research gap, and it has been overtaken by
2026 work. The relevant literature:

- Zhang & Wang (2024), *Enhanced Whale Optimization Algorithm for task scheduling in
  cloud computing environments*, J. Eng. Appl. Sci. 71:121 —
  doi:10.1186/s44147-024-00445-3. **This PDF is already in `../literature/`** as
  `05_EWOA_2024_JEAS.pdf`.
- There is a published *review* of the area: "Use of whale optimization algorithm and
  its variants for cloud task scheduling: a review" (Elsevier book chapter,
  ISBN-linked, `B978-0-323-95365-8.00010-5`), i.e. WOA-for-cloud-scheduling is
  established enough to have review coverage.
- Further 2024–2026 WOA scheduling variants exist (Q-learning hybrids, Lévy-flight
  local search, edge and IoV variants).

A reviewer who opens the candidate's own reference folder finds the contradiction
immediately. The claim is not defensible in any form and should be deleted outright
rather than hedged — see `NOTES_novelty.md` for the replacement wording, which rests on
what was actually measured.

## 3. Reviewer 3's four suggested DOIs — verified

| DOI | Verified title | Year | Venue | Relevant? |
|---|---|---|---|---|
| 10.3390/en17112539 | Real-Time Load Forecasting and Adaptive Control in Smart Grids Using a Hybrid Neuro-Fuzzy Approach | 2024 | Energies | **Tangentially** — ML forecasting of grid quantities |
| 10.3390/electronics13173552 | Innovative Load Forecasting Models and Intelligent Control Strategy for Enhancing Distributed Load Levelling Techniques in Resilient Smart Grids | 2024 | Electronics | **Tangentially** — same |
| 10.32604/cmes.2025.065098 | Real-Time Fault Detection and Isolation in Power Systems for Improved Digital Grid Stability Using an Intelligent Neuro-Fuzzy Logic | 2025 | CMES | **No** |
| 10.3390/en17215412 | Implementation of Fuzzy Logic Scheme for Assessment of Power Transformer Oil Deterioration Using Imprecise Information | 2024 | Energies | **No** |

All four share the same two authors (Fangzong Wang, Zuhaib Nishtar). None concerns
cloud computing, task scheduling, or carbon-aware computing.

**Recommendation.** Cite the two load-forecasting papers in the forecasting
related-work paragraph, where a truthful sentence exists: they are ML forecasting of
grid quantities, which is the same methodological family as forecasting carbon
intensity. Do **not** cite the fault-detection or transformer-oil papers — no honest
sentence connects transformer oil diagnostics to cloud scheduling, and inserting them
would be citation padding that an editor can identify. State briefly and politely in
the response letter that those two fall outside the paper's scope.

Suggested honest framing for the two that are used:

> Machine-learning forecasting of grid-side quantities is an active area: hybrid
> neuro-fuzzy models have been applied to short-term load forecasting and adaptive
> control in smart grids [wang2024load, wang2024levelling]. Carbon intensity is a
> related but distinct target, since it depends on the generation mix rather than
> demand alone, which is why the forecaster here is evaluated against
> carbon-specific seasonal baselines.

## 4. Entries to add to `refs.bib`

All verified. Group A closes the 2023–2026 gap and defines the research gap; group B is
the reviewer's two usable suggestions; group C supports the new experiments.

```bibtex
% ---- A. carbon-aware scheduling: the gap-defining literature (from ../literature/) ----
@article{abbasikhazaei2022energy,author={Abbasi-khazaei, Tahereh and Rezvani, Mohammad Hossein},title={Energy-aware and carbon-efficient VM placement optimization in cloud datacenters using evolutionary computing methods},journal={Soft Computing},year={2022},volume={26},number={19},pages={9287--9322},doi={10.1007/s00500-022-07245-y}}
@article{khodayarseresht2023energy,author={Khodayarseresht, Ehsan and Shameli-Sendi, Alireza and Fournier, Quentin and Dagenais, Michel},title={Energy and carbon-aware initial VM placement in geographically distributed cloud data centers},journal={Sustainable Computing: Informatics and Systems},year={2023},volume={38},pages={100888},doi={10.1016/j.suscom.2023.100888}}
@article{miao2024energy,author={Miao, Zicong and Liu, Lei and others},title={Energy and carbon-aware distributed machine learning tasks scheduling scheme for the multi-renewable energy-based edge-cloud continuum},journal={Science and Technology for Energy Transition},year={2024},volume={79},pages={82},doi={10.2516/stet/2024076}}
@article{danach2026carbon,author={Danach, Kassem and others},title={Carbon-aware scheduling in cloud computing operations: A multi-objective optimisation approach},journal={IET Smart Grid},year={2026},doi={10.1049/stg2.70056}}
@article{ruparel2026carbon,author={Ruparel, and others},title={Carbon-aware scheduling and distributionally robust optimization for cloud systems: forecasting, battery storage, and adaptive ambiguity refinement},journal={Journal of Cloud Computing},year={2026},doi={10.1186/s13677-026-00904-7}}
@inproceedings{moore2025sustainable,author={Moore, and others},title={Sustainable carbon-aware and water-efficient LLM scheduling in geo-distributed cloud datacenters},booktitle={Proceedings of the Great Lakes Symposium on VLSI (GLSVLSI '25)},year={2025},doi={10.1145/3716368.3735301}}

% ---- A2. metaheuristic scheduling 2023-2025: the direct comparators ----
@article{zhang2024ewoa,author={Zhang, Yanfeng and Wang, Jiawei},title={Enhanced Whale Optimization Algorithm for task scheduling in cloud computing environments},journal={Journal of Engineering and Applied Science},year={2024},volume={71},pages={121},doi={10.1186/s44147-024-00445-3}}
@article{feng2025abgwo,author={Feng, Hao and Li, Haoyu and Liu, Yuming and Cao, Kun and Zhou, Xiumin},title={A novel virtual machine placement algorithm based on grey wolf optimization},journal={Journal of Cloud Computing},year={2025},volume={14},pages={7},doi={10.1186/s13677-025-00730-3}}
@article{madhusudhan2023hho,author={Madhusudhan, H. S. and Satish Kumar, T. and Gupta, Punit and McArdle, Gavin},title={A Harris Hawk Optimisation system for energy and resource efficient virtual machine placement in cloud data centers},journal={PLOS ONE},year={2023},volume={18},number={8},pages={e0289156},doi={10.1371/journal.pone.0289156}}
@article{kumar2023eeoa,author={Santhosh Kumar, M. and Karri, Ganesh Reddy},title={EEOA: Cost and energy efficient task scheduling in a cloud-fog framework},journal={Sensors},year={2023},volume={23},number={5},pages={2445},doi={10.3390/s23052445}}

% ---- B. reviewer-suggested, used honestly in the forecasting paragraph ----
@article{wang2024load,author={Wang, Fangzong and Nishtar, Zuhaib},title={Real-time load forecasting and adaptive control in smart grids using a hybrid neuro-fuzzy approach},journal={Energies},year={2024},volume={17},number={11},pages={2539},doi={10.3390/en17112539}}
@article{wang2024levelling,author={Wang, Fangzong and Nishtar, Zuhaib},title={Innovative load forecasting models and intelligent control strategy for enhancing distributed load levelling techniques in resilient smart grids},journal={Electronics},year={2024},volume={13},number={17},pages={3552},doi={10.3390/electronics13173552}}
```

Note on `feng2025abgwo`: the index calls it "ABGWO"; the Crossref title is *A novel
virtual machine placement algorithm based on grey wolf optimization*. Use the Crossref
title, not the index shorthand.

Note on HAPSO (index #1): Crossref resolves only an SSRN preprint
(doi:10.2139/ssrn.5394691) and the journal version appears in press. **Do not cite it
until the version of record exists**, or cite it explicitly as a preprint. This is the
one entry in the index that is not safe to cite as a journal article.

## 5. Where the new citations go in the text

| Location | Change |
|---|---|
| Abstract | delete the "has not previously been used" clause (see `NOTES_novelty.md`) |
| §1 Introduction | add a paragraph engaging 2023–2026 carbon-aware scheduling: `khodayarseresht2023energy`, `miao2024energy`, `danach2026carbon`, `ruparel2026carbon`, `moore2025sustainable` |
| §2.3 Metaheuristics for cloud task scheduling | add `zhang2024ewoa` (and state plainly that WOA has been applied to cloud scheduling), `feng2025abgwo`, `madhusudhan2023hho`, `kumar2023eeoa` |
| §2.4 Carbon-intensity forecasting | add `wang2024load`, `wang2024levelling` with the framing in §3 above |
| §2.5 Summary and research gap | rewrite the gap around what is actually novel: carbon-aware *seeding* as a transferable initialisation strategy, and the regime analysis. Cite `abbasikhazaei2022energy`, `khodayarseresht2023energy`, `miao2024energy` as the carbon-aware anchors that do **not** use seeded initialisation |
| §5 baselines | cite `radovanovic2023carbon` for the VCC baseline and `wiesner2021lets` for the threshold baseline — both are now implemented, so these move from related work to methods |

## 6. Metadata corrections in the existing 19 entries

- `heidari2019harris` — journal is *Future Generation Computer **Systems***; the entry
  reads "Computer Systems" but the correct name is *Future Generation Computer
  Systems*. Verify the rendered string.
- Index #4 (Behera & Sobhanayak, hybrid GA-GWO, doi:10.1016/j.jpdc.2023.104766) —
  online 2023, print 2024. Pick one year and use it consistently.
- Every entry in the added block carries a DOI. Confirm the bibliography style renders
  DOIs; the reviewer explicitly asked for DOI information.

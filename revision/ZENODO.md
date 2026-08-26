# Getting a persistent DOI (Reviewer 2 §5, and R2 #10 reproducibility)

Minting the DOI needs your GitHub and Zenodo accounts, so the final two steps are
yours. Everything Zenodo reads is prepared below.

## Steps

1. Push the repository to GitHub (public).
2. Sign in at https://zenodo.org with GitHub, go to **Account → GitHub**, and flip the
   switch on `carbon-aware-scheduler`.
3. On GitHub, **Releases → Create a new release**, tag `v1.0-revision`.
4. Zenodo archives the release and issues a DOI within a minute or two. Use the
   **Concept DOI** (the "all versions" one) in the paper -- it keeps resolving as you
   add versions.
5. Paste the DOI into the availability statement (`MAINTEX_EDITS.md` §A.7) and the
   `CITATION.cff` below.

`.zenodo.json` and `CITATION.cff` are written into the repository root, so Zenodo picks
up the title, authors, licence and description automatically instead of guessing.

## What gets archived

- `revision/` — 29 experiments, 21 raw per-seed CSVs, the 27-check validation gate,
  both region fetchers, the analysis scripts, and the theory notes
- `notebooks/carbon_aware_scheduling_REVISION.ipynb` — pinned, self-verifying notebook
- `data/carbon/` — five regional carbon traces
- `src/` — the original submitted scripts, unmodified apart from determinism fixes

Two files are deliberately kept and named as failures: `INVALID_raw_E10.csv` (capacity
model that rewarded overloading) and `INVALID_raw_E29_capacity_bug.csv` (overload
measured against M·C instead of real fleet capacity). Keep them. A reproducibility
artefact that shows errors being caught is more credible than one that looks flawless.

## Do not archive

`revision/.env` holds the EIA API key and is gitignored. Confirm with
`git check-ignore -v revision/.env` before pushing. **Regenerate that key** at
https://www.eia.gov/opendata/ once fetching is finished, since it was shared in plain
text.

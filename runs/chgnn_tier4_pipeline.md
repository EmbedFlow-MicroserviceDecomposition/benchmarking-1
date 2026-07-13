# CHGNN × Bearboard (83da66d) + Pharmacy (18acfdf) — full-pipeline cells (10 runs each)

**Date:** 2026-07-14 · **Runner:** Pasindu Ravishan · **Tool:** `tools/chgnn` @ `f94803e`
**Pipeline:** OUR extractor → `adapters/chgnn_adapter.py` (configs prepped
2026-07-13, see commit "Prep Bearboard + Pharmacy cells") → 10 × stock
`Graph.py --model=AE_EGCN_Separate --code=with_edge_loss` → scored with `metrics/`,
both variants + CHM/CHD/BCP. Both are VAE-C lineage datasets
(github.com/rokinmaharjan/{bear-board,pharmacy}).
Configs: `adapters/configs/{bearboard,pharmacy}/`. Raw artifacts:
`runs/chgnn_{bearboard,pharmacy}_repro/`. Scoring script:
`.claude/jobs/87b36165/tmp/score_tier3.py` (same script reused with an updated
`APPS` dict — reads `clusters_N.json` + `outputs/graphs/{app}/{edges,calls}.csv`
for canonical, `result_N.json` through `tools/chgnn/metric.py` for chgnn-repo).

## Run counts

Both apps: n=10, all runs clean (`exit=0` for every run, `driver.log`). No
interruptions this batch.

## Summary (means, n=10)

| dataset | variant | SM | ICP | IFN | NED | BCP | CHM | CHD |
|---------|---------|-----|-----|-----|-----|-----|-----|-----|
| bearboard | canonical | 0.1296 | 0.1211 | 0.50 | 0.7692 | 0.6350 | 0.7604 | 0.7711 |
| bearboard | chgnn-repo | 0.2671 | — | 0.83 | 0.8654 | — | — | — |
| pharmacy | canonical | 0.0916 | 0.3176 | 2.08 | 0.9907 | 1.7819 | 0.5664 | 0.5288 |
| pharmacy | chgnn-repo | 0.1757 | — | 4.63 | 0.9442 | — | — | — |

## Notes and caveats

- **Both apps score much better than the tier-3 cells** (JForum2's SM 0.043 /
  NED 0.03). Both are small, simple CRUD apps (35 and 60 classes) with seed
  counts close to their entrypoint controller counts — consistent with the
  established DayTrader/AcmeAir/JForum2 pattern that CHGNN degrades as
  class-count grows relative to seed-domain count, not an inconsistency.
- **Bearboard:** 3 entity classes referenced only through Spring Data repository
  generics (`MongoRepository<Award,String>`) get no structural edges — the
  extractor doesn't harvest type arguments of external supertypes — excluded via
  `adapters/configs/bearboard/ignore_classes.txt` (known limitation, documented
  at config-prep time, commit "Prep Bearboard + Pharmacy cells").
- BCP use case = web entrypoint (CALL+FIELD reachability); CHM/CHD over
  cross-partition calls from extractor `calls.csv` — same methodology as every
  other full-pipeline cell (`runs/chgnn_tier2_pipeline.md`,
  `runs/chgnn_tier3_pipeline.md`).

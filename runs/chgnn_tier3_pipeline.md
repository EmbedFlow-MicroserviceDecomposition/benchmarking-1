# CHGNN × JForum2 (3724e06) + SpringBlog (07ceda2) — full-pipeline cells

**Date:** 2026-07-13 · **Runner:** Pasindu Ravishan · **Tool:** `tools/chgnn` @ `f94803e`
**Pipeline:** OUR extractor → `adapters/chgnn_adapter.py` → stock
`Graph.py --model=AE_EGCN_Separate --code=with_edge_loss` → scored with `metrics/`,
both variants + CHM/CHD/BCP.
Configs: `adapters/configs/{jforum,springblog}/`. Raw artifacts:
`runs/chgnn_{jforum,springblog}_repro/`. Scoring script:
`.claude/jobs/87b36165/tmp/score_tier3.py` (not checked in — reads `clusters_N.json`
+ `outputs/graphs/{app}/{edges,calls}.csv` for the canonical variant, and
`result_N.json` through `tools/chgnn/metric.py` directly for the chgnn-repo variant).

## Run counts

- **SpringBlog: n=10.** Driver loop completed all 10 planned runs cleanly (exit=0
  each, `driver.log`).
- **JForum2: n=7.** Driver loop (`tier3.sh`, background job `87b36165`) ran 7 clean
  runs (exit=0 each), then run 8 was interrupted mid-training on operator request
  (JForum runs were taking ~1h each and a stopping point was wanted). The driver
  script and its child processes were killed; `result_8.json`/`clusters_8.json`
  never materialized (run 8 exited 252, no `convert_outputs.py` step ran) and were
  removed. **Cells below are means over the 7 completed runs only.**

## Summary (means)

| dataset | n | variant | SM | ICP | IFN | NED | BCP | CHM | CHD |
|---------|---|---------|-----|-----|-----|-----|-----|-----|-----|
| jforum | 7 | canonical | 0.0427 | 0.6339 | 23.95 | 0.0341 | 2.5924 | 0.2233 | 0.0491 |
| jforum | 7 | chgnn-repo | 0.0919 | — | 136.05 | 0.3037 | — | — | — |
| springblog | 10 | canonical | 0.1093 | 0.3638 | 2.48 | 0.9029 | 1.4972 | 0.4779 | 0.3183 |
| springblog | 10 | chgnn-repo | 0.2447 | — | 4.38 | 0.6794 | — | — | — |

(Per-run stdev is in the scoring script's raw output; jforum SM/NED/BCP/CHM/CHD are
tight across runs — stdev ≤0.02 on [0,1]-ish scales — ICP/IFN a bit wider. SpringBlog
CHM/CHD have the widest spread, stdev ~0.16–0.18, consistent with its small op count.)

## Notes and caveats

- **JForum2 quality is poor and this is a real finding, not a bug.** 352 classes
  collapsed into 6 CHGNN clusters (one per seed domain) — SM 0.043, ICP 0.63, IFN
  ~24, NED ~0.03 (almost every class sits in an extreme-sized partition). This
  continues the DayTrader/AcmeAir pattern: CHGNN's seed-driven clustering degrades
  sharply as class count grows relative to the number of seed domains (6 seeds for
  352 classes here, vs. 6 seeds for 25 classes on PetClinic).
  20 of 31 Action classes are entrypoints (install wizard excluded); 89.7% of calls
  resolved by the extractor; 16 classes ignored (cache/exception/util noise,
  `adapters/configs/jforum/ignore_classes.txt`); extractor targets
  `datasets/jforum/src` (Ant-era layout, no `src/main/java`).
- **SpringBlog:** two same-named `PostController` classes (admin vs public) forced a
  fix to `_ep_annotation` to key on the entrypoint's FQN rather than short name —
  otherwise CHGNN's EP feature width would shrink (107 vs 108) and crash. Six
  classes ignored (config/forms with no structural signal,
  `adapters/configs/springblog/ignore_classes.txt`).
- BCP use case = web entrypoint (CALL+FIELD reachability); CHM/CHD over
  cross-partition calls from extractor `calls.csv`, same methodology as
  AcmeAir/PetClinic (`runs/chgnn_tier2_pipeline.md`).

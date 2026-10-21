# AE-K — all 8 datasets, full pipeline (3 runs each)

**Date:** 2026-07-20 · **Runner:** Pasindu Ravishan · **Tool:** `tools/vaec`
(rokinmaharjan/benchmarking-monolith-to-microservices), algorithm ported from
`machine-learning/ae.ipynb` cell-for-cell, unmodified.

**Correction (2026-07-21):** this file/method was originally labeled "VAE-C"
in results.csv. That was wrong: `ae.ipynb`'s `AE` class is a plain
deterministic autoencoder (reconstruction + architecture loss only, no KL
term, no reparameterization/sampling layer), clustered with hard KMeans.
Per Rokin Maharjan's M.S. thesis (Baylor, "Microservice Decomposition and
Message Queues"), that combination is the paper's **AE-K baseline**, not
VAE-C. The real VAE-C (variational encoder + fuzzy c-means) is implemented
separately in `adapters/vaec_c_run.py`, ported from `machine-learning/code.ipynb`
— see `runs/vaec_pipeline.md` for that method's cells. Results.csv rows were
relabeled `method=AE-K` and this file renamed accordingly; the run artifacts
and numbers below are unchanged (only the label was wrong, not the numbers).

AE-K is our third method filled in the matrix (after CHGNN, FoSCI), and the
first with full coverage of every dataset already extracted for CHGNN —
unlike FoSCI, AE-K needs only the static structural graph + entrypoints
(same inputs already produced for the CHGNN cells), no execution traces.

## Pipeline

1. `adapters/vaec_adapter.py` — our `nodes.csv`/`edges.csv` → VAE-C's plain
   adjacency-list JSON (`{className: [dependentClassName, ...]}`), plus the
   entrypoint FQN list (reused directly from `adapters/configs/{app}/entrypoints.txt`,
   already built for the CHGNN cells).
2. `adapters/vaec_run.py` (kept its old filename/module name for git-history continuity; it implements AE-K, not VAE-C) — ports the notebook's algorithm as a standalone
   script:
   - **Entrypoint-reachability matrix E** and **co-entrypoint-path matrix C**
     via DFS from each entrypoint over the adjacency matrix.
   - **Feature matrix** `X = D^-1/2 . A . D^-1/2 . [E | C]` (one GCN-style
     propagation step), standardized.
   - **Autoencoder**: dense tanh layers, dual loss (reconstruction +
     "architecture loss" pulling `Z.Z^T` toward `A`), 8000 full-batch epochs,
     Adam(lr=0.001), 2D embedding — identical architecture/hyperparameters to
     the notebook.
   - **Clustering**: KMeans swept k=2..20 (n_init=10, random_state=0 as in
     the notebook), **k chosen via SSE-elbow** (max perpendicular distance
     from the line connecting the first/last (k,SSE) points). AE-K's own
     paper/notebook doesn't bake in one fixed default k — `result-analyser/`
     folder names vary per app (e.g. `book_no_extra_vae_5_clusters`) — so we
     automate the paper's own stated elbow heuristic rather than hand-pick a
     k per dataset, keeping this consistent with the project's "stock config,
     no manual k-sweep-and-cherry-pick" rule.

**Documented deviation:** the co-entrypoint path DFS (`dfs_for_paths`)
enumerates *all* simple paths from each entrypoint — exponential on graphs
with branching/cycles. Fine for AE-K's own small example apps, infeasible
as-is on our larger datasets (JForum 352 classes, Roller 533 classes). Capped
at 20,000 explored paths/entrypoint (`--max-paths`, prints a WARNING when
triggered — did trigger for JForum and Roller, not for the other 6). Degrades
gracefully (fewer co-occurrence signals) rather than incorrectly; algorithm is
identical to the notebook below the cap.

## Run counts

All 8 datasets × 3 runs each = 24 runs, all clean (`exit=0`, no tracebacks).
JPetStore (24 classes) ≈ 78s/run; Roller (533 classes, largest) ≈ 7min/run —
much faster than expected given the O(n²) architecture-loss term, since 8000
epochs on a tiny per-epoch matrix multiply is still cheap at this scale.

## Summary (means, n=3)

| dataset | SM | ICP | IFN | NED | BCP | CHM | CHD | elbow-k per run |
|---|---|---|---|---|---|---|---|---|
| jpetstore | 0.0807 | 0.4821 | 2.21 | 0.6250 | 0.8934 | 0.3230 | 0.2761 | 6/7/6 |
| spring-petclinic | 0.0495 | 0.7188 | 1.41 | 0.6400 | 0.8305 | 0.6409 | 0.5785 | 7/6/5 |
| bearboard | 0.1803 | 0.3772 | 0.75 | 0.5784 | 0.5927 | 0.8235 | 0.7689 | 7/6/7 |
| springblog | 0.1116 | 0.6287 | 2.87 | 0.7021 | 1.5609 | 0.3915 | 0.2835 | 5/6/7 |
| pharmacy | 0.1252 | 0.6345 | 3.67 | 0.5706 | 1.7392 | 0.2569 | 0.2689 | 5/6/6 |
| acmeair | 0.0955 | 0.3251 | 4.48 | 0.5119 | 1.1118 | 0.3932 | 0.2863 | 7/6/6 |
| jforum | 0.0323 | 0.7539 | 37.72 | 0.0568 | 2.9790 | 0.2343 | 0.0383 | 7/6/6 |
| roller | 0.0127 | 0.7506 | 41.48 | 0.0000 | 3.7015 | 0.2290 | 0.0741 | 6/6/7 |

## Notes and caveats

- **Same large-app degradation pattern as CHGNN and FoSCI**: JForum and
  Roller score dramatically worse than the 6 smaller apps across every
  metric (SM ~0.01-0.03 vs 0.05-0.18, NED ~0-0.06 vs 0.5-0.7, IFN 37-41 vs
  0.75-4.5). This is now a consistent, cross-method finding, not a
  pipeline artifact of any one method — worth flagging as a genuine result
  for the report, not just a caveat.
- **Bearboard and Pharmacy** are AE-K's own author's datasets
  (rokinmaharjan) — the notebook's `entrypoints_texts` dict literally
  hardcodes `bearboard`/`upacharpharmacy` configs, confirming this lineage.
  No published per-metric numbers found in the cloned repo to calibrate
  against directly (no results table, only raw embeddings/weights in
  `result-analyser/`), so these cells are our own run, not a reproduction.
- Elbow-chosen k is stable across runs (5-7 range) for every dataset —
  no wild instability in the model-selection step itself, despite the AE's
  own training being unseeded/stochastic per run.
- No separate "tool-repo" metric variant: AE-K ships no scoring code of
  its own (it's purely an embedding+clustering method), so only the
  canonical variant is recorded, same treatment as any method without a
  bundled `metric.py`.

## Autocare extension (2026-08-18)

Cloned `github.com/rokinmaharjan/autocare-nepal` @ `4850e4c` (the third of
the thesis's own 3 lineage apps, alongside Bearboard/Pharmacy — never
previously cloned into this repo). Spring Boot 2.1.5, Java 8, MongoDB-backed,
48 classes/56 edges after extraction, 4 REST controllers. Ran the extractor
+ `vaec_adapter.py` + `vaec_run.py` (AE-K) with no changes needed — same
straightforward pipeline as bearboard/pharmacy. n=3 stock runs, elbow-chosen
k per run: 6/6/5. **SM=0.1245, ICP=0.3544, IFN=1.60, NED=0.7431, BCP=0.74,
CHM=0.5477, CHD=0.5214.** No published per-metric numbers found (same
situation as bearboard/pharmacy's AE-K rows — the thesis's raw
embeddings/weights don't include a results table for the AE-K baseline).

## Train-Ticket extension (2026-08-18)

Train-Ticket (FudanSELab's benchmark) is NOT a monolith by default — it's
already 41-47 separate deployed microservices, no single input for a
decomposition tool to work on. Rather than guess how the team wanted it
handled, checked how the actual VAE-C-lineage follow-up paper (Sooksatra,
Chy, Arju, Cerny, Rivas, "Using Static Analysis to Aid Monolith to
Microservice System Transformation: Tuning Fuzzy c-Means in a VAE-Based GNN
Approach", ASEW '24, doi.org/10.1145/3691621.3694933) used it: they
**manually converted Train-Ticket into a monolith** (validated via the
app's own end-to-end test suite passing on both versions), then ran their
VAE+FCM decomposition on that monolith and scored the result with SM/IFN
only (same canonical SM formula as ours — no ground-truth/ARI-style ​
comparison to the real 41-service split at all). They published that exact
monolith on Zenodo (zenodo.org/records/11215085) and its source on GitHub
(`github.com/showkat2203/train-ticket-monolith`).

Cloned that repo directly (@ `69ed6e3`, has its own git history) instead of
attempting our own from-scratch 41-service merge — same input the paper
itself uses, no reason to re-derive it. `pom.xml` claims `java.version=1.8`
but pulls `spring-rabbit:3.0.5` (needs Java 17 — a real bug in the paper's
own pom, not investigated further since it's their artifact, not ours);
compiles cleanly under JDK17. Extractor: 285 classes, 390 edges, 40
`@RestController` entrypoints, 100% resolved in the node universe. Ran the
standard non-lineage protocol (elbow-k, n=3 — Train-Ticket isn't one of the
thesis's own fixed-k=5 apps). Elbow-chosen k per run: 7/6/6.

**Result: SM=0.0407, ICP=0.6020, IFN=10.5238, NED=0.1333, BCP=1.9334,
CHM=0.7508, CHD=0.6695.** `published_*` left N/A: the ASEW'24 paper reports
SM/IFN only as sweep curves across (m, threshold) hyperparameter grids
(Figs 4-5), not as single point values directly transcribable into one
cell — see `runs/vaec_pipeline.md`'s Train-Ticket section for the fuller
discussion of that paper's own reported numbers.

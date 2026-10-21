# VAE-C (real) — all 8 datasets, full pipeline

**Date:** 2026-07-21 · **Runner:** Pasindu Ravishan · **Tool:** `tools/vaec`
(rokinmaharjan/benchmarking-monolith-to-microservices), algorithm ported from
`machine-learning/code.ipynb` (NOT `ae.ipynb` — see correction below),
unmodified.

## Correction this file supersedes

An earlier pass ported `machine-learning/ae.ipynb` and labeled the results
"VAE-C" in results.csv. That was wrong: `ae.ipynb`'s `AE` class is a plain
deterministic autoencoder (reconstruction + "architecture loss" only, no KL
term, no reparameterization/sampling layer) clustered with hard KMeans. Per
Rokin Maharjan's M.S. thesis (Baylor University, "Microservice Decomposition
and Message Queues", 2022), that combination is the paper's own **AE-K
baseline** — a comparison point VAE-C is evaluated *against*, not VAE-C
itself. Those 8 cells were relabeled `method=AE-K` in results.csv and their
writeup moved to `runs/ae_k_pipeline.md`. This file and its `VAE-C` rows are
the actual method.

## The real VAE-C

`machine-learning/code.ipynb` (a separate notebook in the same repo) contains
the genuine VAE-C:
- **Variational encoder**: outputs `(z_mean, z_log_var, z)` where
  `z = z_mean + exp(0.5*z_log_var) * epsilon` (reparameterization trick,
  `epsilon ~ N(0,I)`) via a `Sampling` layer — not a direct deterministic
  projection.
- **Loss**: reconstruction + **KL divergence** (`-0.5*sum(1 + log_var -
  mean^2 - exp(log_var))`, standard VAE ELBO term) + the same "architecture
  loss" (`Z.Z^T ≈ A`) as the AE baseline.
- **Clustering**: **fuzzy c-means** (`skfuzzy.cluster.cmeans`, fuzziness
  m=2), not hard KMeans — this is what makes it "VAE-C" (C = fuzzy C-means)
  vs. "AE-K" (K = KMeans) vs. "AE-C" (AE + fuzzy C-means, the third baseline
  in the thesis, which we haven't implemented — no cells need it).
  Membership matrix hardened to a partition via `argmax` over each class's
  cluster-membership scores, matching the notebook's own "no redundancy"
  (non-overlapping) output mode — the thesis's Table 2.3 numbers are from
  this same no-duplication setting (their separate "with duplication"
  experiment, which allows a class in multiple microservices via a
  membership threshold, is out of scope: our metrics module and every other
  method in this matrix assume a hard partition).

## Pipeline

1. `adapters/vaec_adapter.py` — unchanged, reused from the AE-K pipeline
   (`nodes.csv`/`edges.csv` → adjacency-list JSON + entrypoint FQN list).
2. `adapters/vaec_c_run.py` — new script:
   - **Feature construction** (`build_graph`, `build_features`: entrypoint-
     reachability matrix E, co-entrypoint-path matrix C via DFS, GCN-style
     propagation `X = D^-1/2.A.D^-1/2.[E|C]`) — imported directly from
     `adapters/vaec_run.py` (the AE-K script), unchanged. Same documented
     `--max-paths` cap deviation applies (see `runs/ae_k_pipeline.md`).
   - **VAE training**: same dense-tanh encoder/decoder shape as AE-K, 8000
     full-batch epochs, Adam(lr=0.001), 2D embedding — but with the
     `Sampling` layer + KL loss described above.
   - **Fuzzy c-means clustering**: swept k=2..20 (`error=0.005,
     maxiter=1000`, matching the notebook), SSE-elbow selection (same
     `elbow_k` heuristic reused from AE-K, since VAE-C's own repo doesn't
     bake in one fixed default k for datasets outside the thesis's own 3),
     **except** Bearboard/Pharmacy which use a fixed `--k 5` (see below).

## Run protocol: fixed k=5 on the paper's own datasets, elbow-k elsewhere

The thesis (Table 2.1) evaluates VAE-C/AE-C/AE-K on exactly 3 apps —
Bearboard, Autocare, Pharmacy — all partitioned at a **fixed k=5**, chosen
once via an elbow analysis they ran across those datasets, not per-dataset.
Autocare isn't one of our extracted datasets (not cloned); Bearboard and
Pharmacy are. To stay comparable to the thesis's own Table 2.3 on those two
overlapping datasets, we matched their fixed k=5. The thesis also states
VAE-C's own numbers are "the average of five attempts since VAE's latent
space is based on the normal distribution" — so n=5 for these two, matching.

For the other 6 datasets (never in the paper's own evaluation), we kept this
project's standing "no manual k-sweep-and-cherry-pick" rule: automated
SSE-elbow selection, same heuristic as AE-K, n=3 runs (no published n to
match, consistent with the n used for AE-K's non-lineage datasets).

## Run counts

All 28 runs (8 datasets: 2×n=5 fixed-k + 6×n=3 elbow-k) completed cleanly
(`exit=0`, no tracebacks). JPetStore ≈ 80s/run; Roller (533 classes, largest)
≈ several minutes/run at the larger k values fuzzy c-means converged to (7-8).

**Extension (2026-07-30):** DayTrader and PlantsByWebSphere (n=3 each,
elbow-k) were run 2026-07-27 alongside the AE-K/VAE-C dataset-coverage
extension (see the AE-K/VAE-C daytrader/plantsbywebsphere onboarding that
also produced `outputs/graphs/{daytrader,plantsbywebsphere}` for the CARGO
cell), but scoring into `results.csv` was queued behind the CARGO onboarding
and only completed now. Scoring script: `runs/vaec_repro/score_daytrader_pbw.py`.

## Summary (means)

| dataset | n | SM | ICP | IFN | NED | BCP | CHM | CHD | k(s) |
|---|---|---|---|---|---|---|---|---|---|
| jpetstore | 3 | 0.1270 | 0.6513 | 1.94 | 0.4028 | 0.9216 | 0.4822 | 0.3734 | 5/7/6 |
| spring-petclinic | 3 | 0.0716 | 0.6771 | 1.50 | 0.6133 | 0.8318 | 0.5877 | 0.4935 | 6/8/4 |
| bearboard | 5 | 0.1039 | 0.2842 | 0.64 | 0.9118 | 0.6450 | 0.7077 | 0.6829 | 5/5/5/5/5 |
| springblog | 3 | 0.1375 | 0.5232 | 2.33 | 0.8227 | 1.1404 | 0.4245 | 0.4006 | 12/5/5 |
| pharmacy | 5 | 0.0628 | 0.6333 | 3.76 | 1.0000 | 1.4036 | 0.3602 | 0.3553 | 5/5/5/5/5 |
| acmeair | 3 | 0.0935 | 0.4925 | 5.65 | 0.9008 | 0.8508 | 0.3327 | 0.2415 | 6/7/8 |
| jforum | 3 | 0.0226 | 0.7014 | 42.23 | 0.0000 | 2.9718 | 0.2346 | 0.0400 | 6/5/6 |
| roller | 3 | 0.0090 | 0.7863 | 37.68 | 0.0000 | 3.7171 | 0.2261 | 0.0703 | 8/8/7 |
| daytrader | 3 | 0.0522 | 0.8584 | 3.44 | 0.7217 | 2.5815 | 0.4202 | 0.3008 | 10/7/10 |
| plantsbywebsphere | 3 | 0.1497 | 0.7374 | 2.95 | 0.9189 | 0.5119 | 0.4173 | 0.1641 | 7/7/7 |

## Comparison against the thesis's published Table 2.3 (Bearboard, Pharmacy)

| dataset | metric | ours | published (VAE-C) |
|---|---|---|---|
| Bearboard | SM | 0.1039 | **-1.07** |
| Bearboard | NED | 0.9118 | 0.94 |
| Bearboard | IFN | 0.64 | 2.84 |
| Pharmacy | SM | 0.0628 | **-3.79** |
| Pharmacy | NED | 1.0000 | 0.91 |
| Pharmacy | IFN | 3.76 | 7.12 |

**SM is a real, unresolved discrepancy** — the thesis's printed formula
(Eq. in §2, "SM = (1/k)Σ(e_i/n_i²) - (2/(k(k-1)))Σ(e_ij/(2n_i·n_j))") is the
same intra-minus-inter form as our canonical Jin et al. SM, which is always
close to [-1, 1] and typically small-positive for a reasonable partition (our
values across this entire matrix run 0.01–0.18). Their reported values are
large-magnitude and negative (-1.07 to -6.14 across all 3 models/3 datasets
in their table) — inconsistent with their own formula at face value. We did
not attempt to reverse-engineer a different edge-counting convention to force
a match (e.g. summing e_ij across every ordered pair instead of `i<j`, or a
sign flip, neither of which alone explains a shift this large) — flagged
as-is per this project's practice of recording discrepancies rather than
silently "fixing" a metric to match a target number.

NED and IFN are in the right ballpark on both datasets (same order of
magnitude, right direction), which is at least consistent with the algorithm
being implemented correctly even though SM can't be validated this way.

## Notes and caveats

- Same large-app degradation pattern as every other method in the matrix:
  JForum and Roller score far worse than the 6 smaller apps (SM ~0.01-0.02 vs
  0.06-0.14, NED 0 vs 0.4-1.0, IFN 38-42 vs 0.6-5.7).
- SpringBlog's elbow-k landed on 12 once (run 1) vs 5 the other two runs —
  the SSE-elbow heuristic is occasionally unstable on this dataset's SSE
  curve shape; kept as-is (not resampled) per the project's no-cherry-pick
  rule, but flagged in the results.csv note.
- Real VAE-C isn't uniformly "better" than AE-K in our runs (e.g. Bearboard
  SM: AE-K 0.1803 vs VAE-C 0.1039) — the opposite of the thesis's own claim
  that VAE-C outperforms AE-K "in every dataset and every metric". This is
  confounded by AE-K using elbow-selected k (6-7 on Bearboard) vs VAE-C's
  fixed k=5, so it isn't a clean like-for-like comparison; not investigated
  further since resolving it would require re-running AE-K at fixed k=5 too,
  which wasn't asked for.
- No separate "tool-repo" metric variant: VAE-C ships no scoring code of its
  own, only the canonical variant is recorded.

## Autocare extension (2026-08-18)

Autocare is the 3rd of the thesis's own lineage datasets (Table 2.1/2.3:
Bearboard, Autocare, Pharmacy) and had never actually been cloned (noted as
a gap in this file's earlier "Autocare isn't one of our extracted datasets
(not cloned)" line). Cloned `github.com/rokinmaharjan/autocare-nepal` @
`4850e4c`. Ran `vaec_c_run.py --k 5` (matching the thesis's fixed k=5
protocol for this 3-app lineage set) n=5 times, same "average of five
attempts" protocol as bearboard/pharmacy. Chosen k stable at 5 across all 5
runs. **SM=0.0875, ICP=0.3946, IFN=2.16, NED=0.9667, BCP=0.6937, CHM=0.6081,
CHD=0.4802.** `published_*` left N/A in results.csv: unlike bearboard/
pharmacy (whose Table 2.3 values are transcribed in the comparison table
above), Autocare's specific published SM/ICP/IFN/NED numbers were not
available to transcribe here — not fabricated, left blank per project
practice.

## Train-Ticket (2026-08-18)

Train-Ticket is a real 41-47-microservice benchmark (FudanSELab), not a
monolith — no default input for any decomposition method in this matrix.
Investigated how the direct VAE-C-lineage follow-up paper handles this
exact problem: Sooksatra, Chy, Arju, Cerny, Rivas, "Using Static Analysis
to Aid Monolith to Microservice System Transformation: Tuning Fuzzy
c-Means in a VAE-Based GNN Approach" (ASEW '24,
doi.org/10.1145/3691621.3694933) — read via an NSF PAR open-access copy
(par.nsf.gov/servlets/purl/10572044) since the ACM DL copy is paywalled.

**How that paper used Train-Ticket**: manually converted it into a
monolith (42 of the 47 microservices are Java-based; validated the
conversion via the app's own full end-to-end test suite passing
identically on both versions), then ran their own pipeline on it — the
same three-stage shape as our AE-K/VAE-C: static-analysis dependency graph
+ entrypoint-existence/co-existence matrices -> VAE embedding -> fuzzy
c-means clustering, swept over the number of microservices (elbow method,
landed on 20) and the fuzziness coefficient `m` (2 through 10, elbow
picked `m=2`). Published that monolith on Zenodo
(zenodo.org/records/11215085) and GitHub
(`github.com/showkat2203/train-ticket-monolith`) — used directly as our
own dataset clone rather than re-deriving a merge ourselves.

**Their evaluation approach — directly relevant to how we score this
cell too**: they explicitly state *"We do not generate codes for the
resulting microservices. Hence, we cannot run and evaluate it in the
running time. Therefore, we only use metrics that evaluate the
architecture of the microservices for comparison"* — i.e. **no
ground-truth/cluster-similarity comparison against the real 41-service
split at all**, just the same class of standalone structural metrics our
own pipeline already computes. Their SM formula
(`SM = (1/k)Σ(e_i/n_i²) - (2/(k(k-1)))Σ(e_i,j/(2n_i·n_j))`, Section 5.4/
Fig 4) is letter-for-letter our canonical Jin et al. SM — directly
comparable in principle, unlike CARGO's Coupling/Cohesion or FoSCI's
paper-specific formulas noted elsewhere in this matrix. Their IFN
definition ("average external connections per microservice") also matches
ours. They do NOT report NED, ICP, BCP, CHM, or CHD.

**Why `published_*` is still N/A rather than filled from their Fig 4/5**:
they report SM and IFN as sweep curves across a (fuzziness coefficient,
maintainability-threshold) grid (9×5 = 45 points per figure), not a single
reproducible point value — their "recommended" configuration (20
microservices, m=2, threshold=0.25) is visually read off a line chart, not
tabulated as an exact number anywhere in the text. Recording published_*
as N/A rather than eyeballing a chart to a false-precision number, per
project practice of not forcing a match.

Our own run: elbow-k landed on 8/10/8 (not their swept 20 — different
protocol, no k held fixed), n=3 (standard non-lineage protocol).
**SM=0.0167, ICP=0.7799, IFN=10.0417, NED=0.0760, BCP=1.8564, CHM=0.4779,
CHD=0.3479.** Fits the established "large app degrades" pattern already
seen with JForum/Roller (285 classes, SM/NED both near the low end of the
whole matrix) — consistent with, not contradicting, that cross-method
finding.

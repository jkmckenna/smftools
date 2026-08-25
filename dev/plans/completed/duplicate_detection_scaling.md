# Scalable duplicate-read detection

## Why

`reduce_duplicate_reads` (`src/smftools/preprocessing/partitioned_executor.py`) crashed
(silent OOM, no traceback) on a real production experiment
(a ~1.3M-read deaminase run, some `(reference, sample)` groups up to
~46,030 reads). Root cause: `_process_group` (`flag_duplicate_reads.py`) ran a cheap windowed
lexicographic-sort pass (fine, scales well) followed by an uncapped exact hierarchical
clustering pass (`scipy.cluster.hierarchy.linkage` + `pdist`) over every surviving
"representative" read. Most nanopore SMF reads are genuinely distinct molecules, so the
representative count stayed a large fraction of the group size -- `pdist` on ~40,000 points
needs an ~800M-entry condensed distance array (~6.4GB), which is the near-certain crash cause.

Separately, `reduce_duplicate_reads` had zero parallelism and zero memory-watchdog coverage
(unlike every other pipeline stage), and for locus-mode references it `materialize()`d an
entire group's read x site matrix in one call before any per-read algorithm even started -- the
chunking axis was absent, not just under-capped. And a spine.h5ad written *before*
`reduce_duplicate_reads` ran could be left on disk by a crash, silently masquerading as complete
on a later restart (see "Crash-safety fixes" below).

Needs to scale to groups with millions of reads, primarily optimizing memory, secondarily
speed. Approximate (not perfect) duplicate detection is an accepted tradeoff.

## Architecture

### 1. K random-permutation lex-sort banding

`cluster_pass` (`flag_duplicate_reads.py`) generalizes from "forward/reverse of one fixed
column order" to "forward/reverse of `K` independently-seeded random column-order
permutations" (`duplicate_detection_n_permutation_passes`, default 4). `K=0` reproduces the
original behavior exactly. Complexity stays `O(K*N log N + K*N*W)` -- near-linear.

Two reads with true fractional distance `d` share, in expectation, a run of `~1/d` agreeing
columns before the first disagreement under *any* fixed column order. A single order can be
unlucky (an early divergent column separates a true duplicate pair before the windowed
comparison window ever sees them); `K` independent random orders give `K` independent chances,
so the miss probability shrinks geometrically in `K`. Same principle LSH banding formalizes,
implemented as a small generalization of already-tested code instead of a new hash/bucket
structure. MinHash was considered and rejected (approximates Jaccard/set similarity, not the
NaN-aware fractional-Hamming metric this codebase validates against real data).

Deterministic per-pass seeds (`permutation_seed + pass_index`) mean every chunk, in every
round, independently rederives the same `K` orderings -- no shared state needed across workers.

The exact `pdist`+`linkage` hierarchical top-up is kept as a bounded "polish" pass, hard-capped
by `duplicate_detection_hierarchical_max_representatives` (default 5000) -- skipped (logged)
above that, rather than risking the blowup.

### 2. Chunking + multi-round reduction

Added one level above `_process_group`, inside `reduce_duplicate_reads`'s
`reference -> sample -> core_start` loop, *before* `materialize()` runs (see
`duplicate_detection_dispatch.py`). Reads are split into chunks bounded by
`duplicate_detection_max_reads_per_window` (repurposed: no longer raises `MemoryError`, now a
per-chunk ceiling) and the same `target_task_memory_mb` sizing formula `plan_preprocess_tasks`
uses.

- Round 0 chunks are presorted by `duplicate_detection_chunk_presort_metric` (default
  `Fraction_any_C_site_modified`) before splitting -- true duplicates have near-identical
  aggregate metrics, so this front-loads recall into round 0. Rounds 1+ use a freshly reseeded
  random shuffle each round.
- Survivors carried to round N+1 are round-N chunk keepers (`sequence__is_duplicate == False`)
  -- not every read whose local cluster had size 1. A non-keeper's eventual duplicate status is
  already fully determined transitively through the union-find once its keeper participates in
  a later round.
- A single persistent `UnionFind` (owned by `reduce_duplicate_reads`, spanning every read in the
  dataset) receives every round's chunk-local pairs. Union-find composes correctly regardless of
  which chunk/round/pass discovered a pair -- no separate cluster-ID-remapping logic needed.
- Stopping conditions: (a) survivor pool fits in one chunk -> final pass, done; (b)
  `duplicate_detection_min_progress_rounds_before_stop` consecutive rounds add zero new merges;
  (c) hard cap `duplicate_detection_max_rounds` (default 6) -> accept remaining survivors as
  distinct, log a warning, never crash or silently drop reads.
- Chunks returning `None` from `_process_group` (fewer than 2 reads, or no valid comparison
  sites) still carry their reads forward as survivors -- they must not be silently dropped.

Note: with the default `min_progress_rounds_before_stop=1`, a round that happens to find zero
merges stops iteration immediately. This is intentional -- round 0's presort is expected to
catch most real duplicates, and further rounds are a bonus, not a guarantee. It means a
duplicate pair that round 0 fails to co-locate *and* that round 0 finds nothing else to merge
in the same group won't get a second chance under defaults. This is an accepted
recall-vs-compute tradeoff, not a bug (see `test_cross_chunk_duplicate_pair_reconciled_after_round_two`
in `tests/unit/test_duplicate_detection_dispatch.py`, which raises the threshold to prove the
underlying machinery is correct when configured to keep trying).

### 3. Bitpacking

Every read's per-site call array is logically `{0, 1, NaN}` -- never held as a dense
`float32`/`float64` array longer than the single transient step that needs it.
`src/smftools/preprocessing/_bitpack_utils.py`: `pack_calls_and_valid_mask(X_sub)` packs
immediately after `materialize()` returns into two `uint64`-packed arrays (`calls_u64`,
`valid_u64`, ~1 bit/site each); the float array is dropped right after. All windowed
comparisons and the hierarchical top-up's representative-distance checks run via
`popcount_hamming_windowed` (vectorized `popcount((calls_i XOR calls_j) & valid_i & valid_j) /
popcount(valid_i & valid_j)`) instead of per-element float comparison. `unpack_to_float`
reconstructs float `{0.0, 1.0, NaN}` only for small bounded subsets (e.g. capped hierarchical
representatives) that genuinely need float input (PCA, `pdist`).

These helpers were moved out of `preprocess_umi_annotations.py` (which originally defined
`_pack_bool_to_u64`/`_popcount_u64_matrix` for one-hot DNA sequences, no missing-data concept)
into this shared module; that module now imports from here instead.

### 4. Parallelization / memory safety

`duplicate_detection_dispatch.py` dispatches one round's chunk tasks together through the
shared `run_tasks_parallel` (memory-watchdog-covered, broken-pool-retried, same pattern
`execute_partitioned_preprocessing`/`execute_partitioned_spatial` already use) -- chunks within
a round are read-disjoint, so they fan out fully in parallel. Rounds are necessarily sequential
(round N+1's boundaries depend on round N's survivors). The global `UnionFind`/`hamming_minima`
never leave the main process; workers only ever return chunk-local results, folded in by
`_fold_chunk_result_into_union_find`.

## Crash-safety fixes (prerequisite, same investigation)

Two robustness bugs found while root-causing the crash, fixed independently of the scaling work:

1. **Staging write** (`execute_partitioned_preprocessing`): `spine.h5ad` used to be written to
   its *final* path before `reduce_duplicate_reads` ran, then written again after. A crash
   between those two writes left a pre-dedup `spine.h5ad` at the final path, indistinguishable
   from a completed one. Now written to a `.partial` staging path first; the final path only
   ever receives the fully-merged (QC + dedup) spine.
2. **Skip-if-exists completeness check** (`cli/preprocess_adata.py`): the "skip preprocessing,
   partitioned spine found" restart check only tested file existence. Now also verifies
   `passes_qc`/`passes_dedup` are actually present in the spine's `obs` before trusting it --
   otherwise re-runs preprocessing instead of silently propagating unfiltered, non-dedup'd data
   to every downstream stage with no error anywhere in the chain (exactly what happened on the
   that run: spatial and HMM both analyzed the full unfiltered read set).

## Config

See `duplicate_detection_*` fields in `src/smftools/config/default.yaml` and
`src/smftools/config/experiment_config.py` for the full new/changed field list and defaults.

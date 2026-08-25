# Audit: I/O, memory, and parallelism scaling in `smftools experiment full`

> **Repository state reviewed:** `107c668` — resolved from the `v2.1.0` version this document names.
> **538 commits on `main` since.** An audit describes the code at a moment; it
> goes stale rather than completing. Re-verify any specific claim before relying on it.

Scope: the current (v2.1.0, branch `2.1.0-project-catalog`) partitioned-store
pipeline -- `raw → preprocess → spatial → hmm`. Supersedes `load_preprocess_
audit.md`, which describes the pre-partitioned, monolithic-`.h5ad.gz` architecture
(branch `0.4.5-layer-audit`) that no longer exists; not edited in place since it's
a materially different system, not an update to the same one.

Question asked: does this scale to millions-to-tens-of-millions of reads and to
genome-wide references, tunable to whatever machine it runs on? Findings below are
evidence-based (file:line references, verified against the actual code, not
assumed from architecture docs) and organized by where in the pipeline they bite.

## Executive summary, ranked by impact

1. **No parallelism at all across preprocess/spatial/hmm tasks** -- the single
   biggest gap relative to the stated goal. Every task-based stage runs its
   independent, already-memory-bounded units of work strictly sequentially, on one
   core, regardless of `cfg.threads`. This is the most consequential finding
   because it's also the cheapest to fix (the tasks are already isolated,
   independent, and file-addressed -- built for parallelism, just not wired to it).
   **Done for raw ingestion (`2.4.0-pysam-direct-signal-backend`) and preprocess/
   spatial/hmm (`2.5.0-parallel-task-execution`).** Raw ingestion's extraction loop
   (across `conversion`/`deaminase`/`direct` pysam backend) splits each
   reference's reads into `cfg.threads` read-count-balanced buckets and extracts
   concurrently via `ProcessPoolExecutor`. Two real, non-obvious findings from
   getting this right, both worth remembering for any future work that splits
   reads by genomic position: (a) equal-*width* genomic windows are badly
   imbalanced on real amplicon data -- reads cluster at only a couple of exact
   positions (PCR/library duplication near a primer site), so one window can end
   up with the large majority of a reference's reads regardless of window count;
   (b) splitting by read *identity* (round-robin over read_ids, then filtering via
   `extract_read_relative_base_identities`'s existing `read_name_filter` param)
   sidesteps this entirely and gives exact balance regardless of position.
   Preprocess/spatial/hmm task dispatch (`preprocessing/partitioned_executor.py`,
   `tools/partitioned_spatial.py`, `tools/partitioned_hmm.py`) now goes through a
   new shared orchestrator, `memory_guard.run_tasks_parallel`, instead of a plain
   list comprehension.

   Raw ingestion's initial real-data run only showed a modest ~1.5x speedup from 8
   workers (not the hoped-for 6-8x) with the root cause left unprofiled at the
   time -- **that root cause was found while parallelizing preprocess/spatial**:
   task-level (process-count) parallelism was compounding with numpy's own BLAS
   thread pool, each worker independently multi-threading its own numpy calls
   across every physical core (confirmed: 8 concurrent spatial task workers each
   independently at 200%+ CPU, ~734% total on an 8-core-class budget -- severe
   oversubscription that makes wall-clock time *worse*, not better). Fixed by
   forcing single-threaded BLAS/OMP/numexpr as the literal first statements in
   `smftools/__init__.py` -- has to be there, not `cli_entry.py`: Python always
   fully runs a package's `__init__` before any of its submodules, and
   `__init__.py` itself already imports pandas/numpy transitively, so anything in
   `cli_entry.py` was already too late. Per-worker-pool `ProcessPoolExecutor`
   initializers were tried first and found unreliable specifically because this
   codebase's `forkserver` start method pre-imports the whole app (numpy
   included) into a warm template process before any pool or its initializer
   exists -- env var changes after that point don't retroactively shrink an
   already-initialized BLAS thread pool. After the fix, real-data spatial timing
   went from oversubscribed/~734% CPU to a healthy ~60% CPU and 5m16s -> 2m02s
   wall-clock (preprocess and spatial both verified byte-identical output,
   sequential vs. parallel, on the same real dataset). Raw ingestion's own timing
   wasn't re-measured after this fix landed -- likely also benefits, not
   confirmed.

   **A second real crash, unrelated to BLAS**: HMM tasks resolve a torch device
   per task (`resolve_torch_device`); multiple worker *processes* concurrently
   initializing/using the same GPU context (MPS on Apple Silicon, confirmed via a
   real HMM run; CUDA not independently verified either) reliably crashed the
   whole pool (`BrokenProcessPool`). `run_tasks_parallel` gained a
   `force_sequential` param; `execute_partitioned_hmm` resolves the device once
   upfront and forces sequential execution whenever it isn't `"cpu"`. This is a
   genuinely different failure mode from BLAS oversubscription -- BLAS threads
   within one process don't fight over a single shared GPU context the way
   independent processes do -- so the fix is a targeted opt-out, not a variant of
   the BLAS fix.

   **New config**: `max_memory_percent` (default 60.0) / `max_memory_gb`
   (optional, more restrictive of the two wins when both set) bound the whole
   workflow's aggregate memory use, not per-task. Enforced via a Linux cgroup v2
   cap (`memory_guard.enable_aggregate_memory_cap`, now config-driven, wired in
   once per CLI invocation via `cli/helpers.py::load_experiment_config`) or, on
   macOS, proactive worker-count capping (`resolve_max_workers`) plus a reactive
   per-worker RSS watchdog (`start_worker_watchdog`, generalized to support both
   `multiprocessing.Pool` and `ProcessPoolExecutor`). Not yet stress-tested under
   real memory pressure -- on the 128GB test machine used this session, the
   default 60% budget (76.8GB) was never close to binding (`cfg.threads=8` was
   always the limiting factor; observed peak RSS was ~4GB for a single HMM task).
   The reactive watchdog's kill mechanism *is* directly verified (an
   intentionally-overbudget worker gets killed within seconds, both pool types),
   just not the proactive worker-count throttling under a genuinely tight budget
   on a real workflow run.
2. **Raw ingestion builds one full-experiment DataFrame in memory before writing
   anything.** All read-level ragged data (sequence/quality/mismatch/modification
   arrays, one row per read) is assembled into a single `pd.DataFrame` across every
   reference before `write_raw_store` writes a single shard. This is the one place
   in the pipeline where memory scales with *total experiment size*, not with a
   configured bound -- everywhere else, the architecture already avoids that.
   **Addressed for `conversion`/`deaminase` modality (`2.3.0-streaming-raw-
   ingestion`)**: `cli/raw_adata.py::build_ragged_records_streaming` extracts and
   yields one reference's frame at a time instead of accumulating every read's row
   across the whole experiment; `informatics/raw_store.py::write_raw_store_
   streaming`/`_write_raw_shards_streaming` consume the generator and never hold
   more than one reference's ragged array data resident, writing+freeing shards
   before the next reference arrives. A per-reference-group global-sort
   equivalence argument (`Reference_strand` is the outermost sort key) guarantees
   byte-identical shard/catalog/`canonical_row` output to the whole-frame writer.
   Deaminase modality needed an extra fix: a single FASTA record/chromosome can
   yield reads split across `_top`/`_bottom` `Reference_strand` (decided per-read
   by mismatch trend, not per-chromosome), so `_split_by_reference_strand` sub-
   groups each chromosome's extracted frame before yielding -- caught by real-data
   verification (a first pass silently mislabeled ~45% of reads under the wrong
   Reference_strand: 4 groups produced instead of the expected 8). Verified
   byte-identical `Reference_strand` assignment across all 7,321 reads against the
   already-validated non-streaming run. **`direct` modality addressed
   (`2.4.0-pysam-direct-signal-backend`)**: a new config toggle,
   `direct_signal_backend` (default `"pysam"`, `"modkit"` for the original
   TSV-based path), decodes each read's own MM/ML BAM tags directly via
   `pysam.AlignedSegment.modified_bases` instead of joining a modkit-extract
   TSV -- verified empirically against a real modkit TSV that this reproduces
   modkit's `call_prob`/`call_code` exactly (0 mismatches) for every explicitly-
   called position, and correctly leaves uncalled positions `NaN` rather than
   modkit's synthetic "canonical, prob=1.0" fill (a correctness improvement,
   not a divergence -- the existing TSV-based path was already treating that
   fill as if it were a real call). This removes the external `modkit extract`
   subprocess and its whole-file TSV entirely for the default backend, so
   `direct` modality now streams like conversion/deaminase too (`modkit`
   backend keeps the original whole-frame path, since its TSV join still isn't
   streaming-compatible). See finding 1 below for the parallelism layered on
   top of this.
3. **Genome-mode task count scales with reference length ÷ a fixed 10kb tile,
   with no read-density awareness** -- a genome-wide reference produces hundreds of
   thousands of tasks/catalog rows regardless of how much data is actually in each
   window, and task-planning cost is a repeated full linear scan per window.
4. **The per-task memory budget (`target_task_memory_mb`) undercounts real peak
   memory by ~4-6x** -- it models one dense array, but a task's execution creates
   several more (binarized copy, NaN-fill layers, `_core_result` copies) that
   aren't counted. **Addressed (`2.2.0-incremental-layer-writes`)**: preprocess
   tasks now stream each derived layer to its zarr store via
   `informatics/incremental_zarr.py::append_zarr_layer` and free it immediately
   (`preprocessing/partitioned_executor.py::execute_preprocess_task`), instead of
   holding every layer in memory until one combined write, cutting peak per-task
   residency from ~9-10 full-size arrays (worst case) to ~2-3. Modality-aware:
   `direct`'s binarized/Youden layer is `clean_NaN`'s read source, so it's kept
   resident through `clean_NaN` and only written afterward; `conversion`/
   `deaminase` have no such extra layer. Verified byte-identical output (X, every
   derived layer, var stats) against the pre-refactor run on the same real 7,321-
   read deaminase experiment used to validate Phases 3-4, across all 107 tasks.
5. **No host-resource detection anywhere** -- `target_task_memory_mb`,
   `genome_tile_size`, `genome_tile_halo`, and `threads` are all static config
   values with hardcoded defaults; nothing probes available RAM/CPU count to sanity
   -check or auto-scale them.
6. **Catalog re-reads and full ragged CIGAR reconstruction are both uncached**,
   paid from scratch on every `materialize()` call that doesn't hit the dense/fast
   path -- multiplicative with the genome-mode task-count problem in (3).
7. **HMM EM fitting had a real correctness bug, not just a speed problem --
   **done (`2.7.0`-`2.8.0`).** Investigating "should `hmm_max_iter` be lower?"
   surfaced that `fit_em`'s log-likelihood proxy was mathematically ~0 at every
   iteration by construction (computed from an already-normalized posterior),
   so the tolerance-based early stop fired at iteration 2 on essentially every
   real fit -- existing models may have been undertrained, not slow to
   converge. Fixed the log-likelihood computation, switched the convergence
   tolerance to relative (a fixed absolute epsilon isn't portable across
   log-likelihood magnitudes, which scale with dataset size), and eliminated a
   redundant duplicate forward-backward pass found while fixing this (~35%
   per-iteration wall-clock reduction, verified numerically identical to the
   pre-fix code). `torch.compile` was benchmarked on the position loop and
   rejected -- 195s one-time compile cost and ~19x *slower* per-call than
   eager on real data. See finding 3b for the full account.

## 1. Raw ingestion (`smftools experiment raw`)

**The core risk:** `cli/raw_adata.py::build_ragged_records` (`raw_adata.py:334-379`)
loops every reference, calls `extract_read_relative_base_identities`
(`bam_functions.py:4019-4086`, itself building a full per-chromosome `records = []`
list before returning), and accumulates every read's row into one Python list
across the *entire* experiment before `pd.DataFrame(rows)` is built
(`raw_adata.py:379`). `write_raw_store` is only called after this whole frame
exists (`load_adata.py:1104-1112`). Ragged (query-length, not reference-length)
arrays keep this from being the worst-case `n_reads × genome_length` blowup, but
it's still O(total sequenced bases) resident in one object before any bounded,
shard-sized write occurs.

`write_raw_store` itself compounds this: `validate_ragged_frame` full-copies the
frame (`ragged_store.py:196`), and `sort_values` on the complete frame
(`raw_store.py:219`) can leave `frame`/`normalized`/`sorted_frame` all alive at
once -- roughly 3x peak before the actual shard-writing loop
(`raw_store.py:221-274`, bounded by `shard_size`, default 100k reads) ever starts.
CIGAR validation also runs via `frame.iterrows()` (`ragged_store.py:208`) --
single-threaded, once on the full frame and again per shard -- so every read's
CIGAR is parsed twice, in a Python loop, not vectorized.

`get_native_references` (`fasta_functions.py:284-288`) also holds every
chromosome's full sequence as a Python string for the whole run; `conversion`/
`deaminase` modality additionally caches a per-chromosome complement dict
(`raw_adata.py:339-350`). Fine for amplicon-scale FASTAs; for genome-wide
references this is a fixed but non-trivial (multi-GB for a mammalian genome)
memory floor held for the entire raw stage, with no windowed FASTA reading.

**Already scales fine:** shard writing itself (bounded, `shard_size`-driven),
catalog/molecules/barcode-index writes (one-shot, not per-read appends),
`storage_planner.py`'s locus-vs-genome auto-threshold (`max_full_matrix_gb`,
default 8 GB) for the *downstream* dense-cache decision.

**Already tunable / already parallel:** `shard_size`, `start_bin_size`,
`analysis_mode`, `max_full_matrix_gb`, `genome_tile_size`/`genome_tile_halo`.
`cfg.threads` genuinely parallelizes minimap2/dorado, samtools sort/index, UMI
extraction (`ProcessPoolExecutor`, `bam_functions.py:962-1068`), barcode
extraction (same pattern), POD5 metadata reads, and BAM QC (`ThreadPoolExecutor`).
**Update (`2.4.0-pysam-direct-signal-backend`)**: the streaming builders
(`build_ragged_records_streaming`, covering `conversion`/`deaminase`/`direct`
with the pysam backend) now do parallelize `extract_read_relative_base_
identities` across `cfg.threads`, via read-count-balanced buckets (see the
executive summary's finding 1 for the design and its caveats). The non-
streaming `build_ragged_records` (only reachable now via `direct` modality's
`modkit` backend) remains single-threaded, as does `validate_ragged_frame`'s
row loop in both paths.

## 2. Task planning (`dispatch_plan.py::plan_preprocess_tasks`)

The memory-budget formula: `memory_budget = target_task_memory_mb * 1024**2`;
`reads_per_chunk = memory_budget // (loaded_width * BYTES_PER_WORKING_POSITION)`
(`dispatch_plan.py:107,134-137`), with `BYTES_PER_WORKING_POSITION = 8`
(`dispatch_plan.py:13`, undocumented). By its use, this models exactly **one**
dense `n_reads × width` array -- i.e. `X` alone. It is never scaled up for the
other full-size arrays a task actually holds simultaneously during execution:
`binarize_adata`'s `X_bin = X.copy()` (a second full copy), the default 2 (up to
5, if configured) NaN-fill layers (`clean_NaN.py:15,17-18`), and `_core_result`'s
own fresh copies of every derived layer (`partitioned_executor.py:290-303`). Real
peak per task is roughly 4-6x the configured budget under default settings --
`target_task_memory_mb` is closer to "one array's budget" than "task memory
budget" as named.

Genome-mode tiling (`reference_windows`, `dispatch_plan.py:71-79`) uses a fixed
`genome_tile_size` (default 10,000 bp, `experiment_config.py:1087`/`default.yaml:
518`) regardless of reference length: `n_windows = ceil(length / tile_size)`. A
3.1 Gb genome produces ~310,000 windows per reference-strand before barcode/chunk
splitting multiplies that further. There's no read-density-aware sizing -- a
window's `reads_per_chunk` comes purely from window *width*, never from how many
reads actually land there, so a dense/duplicated locus just produces more
same-size chunks, not a smarter split.

Planning cost itself: for each window, `overlapping = reference_obs.loc[...]`
(`dispatch_plan.py:117-120`) is a full boolean-mask scan of that reference's obs
rows -- `O(n_windows × n_reads_for_that_reference)` total, no spatial index or
sort-and-bisect. At genome scale (many windows) times high read counts, this
linear-scan-per-window cost is paid once up front for the whole plan, before any
task executes.

**Already scales fine:** `plan_preprocess_tasks` only ever touches `spine.obs`
(thin, metadata-only, no signal arrays) -- a real concern at tens of millions of
reads is DataFrame size, not O(reads×positions) data. Per-task chunking itself
(`_chunks`) is O(n) and deterministically bounds reads/task.

**Already tunable:** `target_task_memory_mb` (default 512 MB), `genome_tile_size`/
`genome_tile_halo` (10,000/1,000), `max_full_matrix_gb` (8.0). **Not** tunable or
detected: none of `target_task_memory_mb`/`genome_tile_size`/`genome_tile_halo`/
`threads` are derived from or checked against actual host RAM or CPU count --
`psutil`/`os.cpu_count`-based detection doesn't exist anywhere in this path.

## 3. Task execution -- no parallelism -- **done (`2.5.0-parallel-task-execution`)**

(found directly, both preprocess and hmm and spatial confirmed identical)

```python
# preprocessing/partitioned_executor.py:945 (as found -- now goes through run_tasks_parallel)
records = [execute_preprocess_task(spine_path, task, cfg, output_dir, ...) for task in task_list]
# tools/partitioned_hmm.py:455 (as found -- now goes through run_tasks_parallel)
records = [execute_hmm_task(spine_path, task, cfg, output_dir, models_dir) for task in tasks]
# tools/partitioned_spatial.py:1341 (as found -- now goes through run_tasks_parallel)
records = [execute_spatial_task(spine_path, task, cfg, output_dir) for task in tasks]
```

All three were plain sequential list comprehensions; `cfg.threads` was never
consulted (confirmed by grep at the time). Now all three go through
`memory_guard.run_tasks_parallel` -- see the executive summary's finding 1 for
the full account of what that took to get right (BLAS oversubscription, GPU
context sharing, the new memory-budget config). No architectural change to the
tasks themselves was needed: they were already independent, memory-bounded, and
file-addressed, exactly as this finding predicted.

This was the sharpest finding in the audit: the entire task-planning apparatus
(bounded memory, `(reference, window, barcode, chunk)`-addressed, file-isolated
shards, no shared mutable state between tasks) was *already shaped* for
embarrassing parallelism -- there was no architectural reason these loops
couldn't be `Parallel(n_jobs=cfg.threads)`-wrapped or run through a process
pool. A genome-wide run with hundreds of thousands of tasks still executing them
one at a time on a single core was very likely the dominant wall-clock
bottleneck for any large run, ahead of
every I/O finding below.

## 3b. HMM EM fit correctness and efficiency -- **done (`2.7.0`-`2.8.0`)**

Started as "should we lower `hmm_max_iter`?" (following up on finding 1's HMM
timing question). Real-data convergence probing on
a large multi-reference array run surfaced a genuine correctness bug before
any tuning question could even be answered:

- **`fit_em`'s `ll_proxy` was not a log-likelihood at all** (`hmm/HMM.py`, all
  three concrete implementations). It was computed as
  `sum(log(gamma.sum(dim=2)))`, but `gamma` is already row-normalized by
  `_forward_backward` (`gamma.sum(dim=2) == 1` at every iteration by
  construction) -- so `ll_proxy` was ~0 always, independent of fit quality.
  Confirmed on real data: forcing 150 unbounded iterations, consecutive
  `ll_proxy` diffs never exceeded ~1.6e-8. With the old default
  `hmm_tol=1e-4` (absolute), this meant the EM loop's early-stop check fired
  at iteration 2 for effectively every production fit -- `hmm_max_iter=50`
  was likely never actually reached, and existing fitted models may have
  been undertrained. **Fixed** (`2.8.0-hmm-loglikelihood-fix`) by computing
  the true log-likelihood from the forward pass's terminal normalizer
  (`alpha[:, L-1, :].logsumexp()`) instead.
- **The convergence tolerance needed to be relative, not absolute.** A real
  log-likelihood's magnitude scales with dataset size (N*L), so a fixed
  epsilon isn't portable: on the largest real per-sample task (472 reads),
  the old absolute `1e-4` didn't fire until iteration ~102, while >99.99% of
  the achievable log-likelihood improvement was already captured by
  iteration ~29 on both a large (472-read) and small (97-read) representative
  task. Switched the break condition to `tol * max(abs(current_ll), 1.0)`
  and lowered the default to `1e-5` (matches the empirical ~29-iteration
  convergence point). `hmm_max_iter` itself was left at 50 as a safety cap --
  no evidence it needs to go lower now that the tolerance actually works.
- **`fit_em` was computing the forward-backward recursion twice per EM
  iteration.** It called `_forward_backward` once for `gamma`, then manually
  recomputed alpha/beta from scratch a second time for the expected-
  transition counts (`xi`) -- 4*L sequential Python-loop steps per iteration
  where only 2*L are needed. Factored the shared alpha/beta pass into
  `BaseHMM._alpha_beta` (and a per-bin variant for the distance-aware
  architecture), used by both `_forward_backward` and `fit_em`. Verified
  numerically identical to the pre-refactor code (hist/emission/transition
  match to 1e-9 with early-stopping disabled, isolating this to a pure
  performance change). Measured ~35% wall-clock reduction per EM iteration on
  the same real 472-read task (192ms/iter -> 124ms/iter). Combined with the
  tolerance fix (now typically stopping around iteration ~29 instead of
  running all 50), real per-task fit time drops roughly 2.5-3x end to end.
- **`torch.compile` was benchmarked and rejected.** Compiling
  `BaseHMM._alpha_beta` (the sequential position loop) on real data: one-time
  warmup/compile cost was **195s** (vs. the entire eager 30-iteration fit
  taking 3.7s), and post-warmup per-call time was ~19x *slower* than eager
  (2.37s/call vs. 0.12s/call). The loop's per-step tensor ops are tiny (2-3
  states), so `torch.compile`'s tracing/guard overhead for a data-dependent-
  length Python loop dominates; not worth adopting. A parallel/associative-
  scan reformulation (O(log L) sequential depth) was considered but not
  attempted -- given L is already only ~thousands of positions and K is 2-3
  states, it's unlikely to justify the implementation/numerical-stability
  risk at the current data scale.
- **Fit-history is now persisted and plotted.** Every `.pt` checkpoint saved
  by `HMMTrainer._save` carries its `fit_em` run's `hist` (per-iteration
  log-likelihood) under `fit_history`. `tools/partitioned_hmm.py`'s
  `_plot_hmm_fit_history` reads these straight from disk (no refit needed)
  and plots one convergence curve per model into the previously-empty
  `training` plot category (`<hmm_output>/plots/training/hmm_fit_history.png`,
  `plot_type="hmm_fit_history"`).

## 4. The shared read path (`materialize()`) and project-level pooling

Every `materialize()` call that isn't served by today's new preprocess-X fast path
(single covering shard, exact-match selection -- see `informatics/experiment_
storage_schema.md` Phase 4) re-reads the relevant catalog parquet(s) from scratch,
with no caching anywhere in `partition_read.py` (confirmed by grep -- no
`lru_cache`/memoization): `_derived_layer_names`, `_load_preprocess_x_selection`,
`_load_tiled_cache_selection`, `_overlay_preprocess_var`, and
`_overlay_spatial_read_metrics` each independently call `pd.read_parquet` on their
respective catalog. `_overlay_preprocess_layers` is the worst case -- it re-reads
`preprocess_catalog`/`hmm_catalog` from scratch **inside a nested loop over
references**, so one `materialize()` call spanning several references pays that
read multiple times in a single call. A caller that loops `materialize()` per
window/barcode (`cli/stage_input.py::iter_stage_slices`, used by preprocess's
`_read_span_quality_plots` diagnostic and others) additionally re-does a full,
non-lazy `anndata.read_h5ad` of the whole spine on every single iteration
(`_resolve_spine` → `load_spine` → `safe_read_h5ad`).

At genome scale, task catalogs are large (see §2's ~310,000-window estimate,
multiplied by barcode count) -- this makes "re-read the full catalog every time"
a real, multiplying cost on top of the per-call CIGAR reconstruction cost below,
not just a rounding error.

Ragged reconstruction (`ragged_store.py::materialize_ragged`, used whenever a
selection isn't served by a dense/tiled/fast-path shard) is not vectorized:
`frame.iterrows()` (`ragged_store.py:419`) plus a pure-Python CIGAR-pair generator
(`iter_cigar_aligned_pairs`) with scalar numpy indexing per aligned base. Cost is
O(reads × aligned positions) in a Python loop, re-paid from scratch on every call
with no per-process memoization.

Project-level pooling: `set_store.iter_set_parts` is a genuine generator (bounded
memory regardless of set size) and the new preprocess-X fast path already avoids
ragged reconstruction when it hits. Two real gaps: `project_adata` appends every
materialized part to a Python list before the 8 GiB guardrail check runs per-append
(`catalog.py:253-286` -- the check happens *after* appending, so peak can briefly
exceed the limit by one part's worth), and `embedding_store.fit_or_extend_
embedding` calls `project_adata(..., allow_large=True)` **unconditionally**
(`embedding_store.py:264`) -- every embedding fit/extend bypasses the size
guardrail entirely, and "extend" still re-materializes and re-featurizes (with a
float64 upcast) the *entire* pooled set before slicing out only the new rows.

## Recommendations, roughly in priority order

1. **Done (`2.5.0-parallel-task-execution`)** -- see finding 3/the executive
   summary's finding 1 for the full account. Used a `ProcessPoolExecutor`
   (`memory_guard.run_tasks_parallel`), not `joblib.Parallel` as originally
   suggested here -- needed more control than joblib exposes (a custom worker
   initializer for the BLAS-thread fix, a watchdog wired around the pool, an
   opt-out for HMM's GPU-sharing constraint) and no task semantics needed to
   change either way.
2. **Done for `conversion`/`deaminase` (`2.3.0-streaming-raw-ingestion`) and
   `direct` (`2.4.0-pysam-direct-signal-backend`, default backend)** -- see
   finding 2's update above. Turned out reimplementing MM/ML decoding was
   *not* out of scope after all, once actually checked empirically against
   real data: `pysam.AlignedSegment.modified_bases` is a standard, spec-
   compliant decoder already available (not something to newly build), and
   dorado (not modkit) is what computes the underlying probabilities in the
   first place -- modkit is "just" a formatter over the same tag data.
   Verified this by comparing modkit's own TSV output against a direct
   pysam decode on the same real reads (0 mismatches). This resolved the
   TSV-ordering question from the original research too, since it's now
   moot -- the pysam backend never touches the TSV at all. The `modkit`
   backend (`direct_signal_backend="modkit"`) is kept for parity/fallback
   but still has this recommendation's original gap (whole-file TSV, no
   `chunksize`); not addressed, since it's no longer the default path.
3. **Done (`2.2.0-incremental-layer-writes`).** Rather than better-estimating
   the old batched-write peak, this was addressed by removing most of the peak
   directly -- see finding 4's update above. `target_task_memory_mb`'s formula
   itself is unchanged (still models one array), but the gap between "budgeted"
   and "actual peak" it was flagging is now much smaller (~2-3x instead of
   ~4-6x), since layers no longer accumulate. Re-deriving the formula to be
   fully accurate is lower priority now that the dominant multiplier is gone.
4. **Add read-density awareness to genome-mode tiling**, or at minimum make
   `genome_tile_size` scale-aware (e.g. derived from reference length so a 3 Gb
   genome doesn't default to the same 10 kb tiles tuned for amplicon-scale use).
5. **Cache catalog reads within a single `materialize()` call / short-lived
   process scope.** Even a simple `functools.lru_cache` keyed on resolved catalog
   path (invalidated by mtime, or just process-lifetime since these are
   append-only per run) would collapse the 5-7 redundant reads per call and the
   nested-loop re-read in `_overlay_preprocess_layers`.
6. **Add host-resource detection as a sanity check, not a silent override** --
   e.g. warn (not auto-change) when `target_task_memory_mb * (expected parallel
   workers)` exceeds detected available RAM, or when `threads` exceeds
   `os.cpu_count()`. Keeps the "tunable to available threads and memory" goal
   honest without taking away explicit user control.
7. **Fix `embedding_store`'s unconditional `allow_large=True`** and move
   `project_adata`'s guardrail check to run before appending a part that would
   cross the limit, not after.

None of this blocks correctness at small/medium scale -- everything above was
found by reading the code, not by the real-data test run in this session (7,321
reads, 8 references, ~89 spatial tasks), which completed cleanly end-to-end. These
are the specific places that will start to matter as read count and reference
scope grow toward the stated millions-of-reads / genome-wide target.

---

# Session update (2026-07-19): first real memory-pressure stress test

The 11-experiment `Nkg2a_DAFseq_merged_v2` batch (~0.5-1.0M reads/experiment, up
to 6 `Reference_strand`s each) was the first run big enough to actually bind the
memory machinery that finding 1 flagged as "not yet stress-tested under real
memory pressure." It bound hard, in several distinct places -- most of which the
audit above had either half-anticipated or not seen at all. What follows is what
broke, what was fixed, and what is newly flagged.

## A. Raw ingestion re-blew-up in the *parent* despite the streaming writer -- **fixed**

Finding 2 (streaming raw ingestion) prevented the parent from holding the whole
experiment's ragged frame, but the real run still spiked the parent to ~90GB
(masked by macOS's memory compressor showing only ~500MB "free" -- the compressor
buys runway, it does not bound growth). Two stacked causes, both parent-side and
both invisible to the per-*worker* watchdog:

1. `_ChromosomeGroupAccumulator` accumulated a whole chromosome's ragged data
   before writing (it existed to prevent a shard-index collision bug, and did that
   correctly, but held the whole chromosome to do it). Fixed by splitting it into
   `add_partial`/`complete` with a `raw_shard_flush_max_reads` threshold (default
   20000) that flushes bounded batches mid-chromosome. Persistent per-`(reference,
   start_bin)` shard-index tracking preserves the no-collision guarantee across
   partial flushes. (commit `e487f9a`)
2. The real culprit: `_map_references_parallel` submitted **every** bucket task for
   the whole experiment to the pool up front (`{pool.submit(...): args for args in
   items}`), fully decoupled from `max_workers`. Fast workers producing large
   per-bucket ragged frames + a single-threaded consumer draining them = completed
   results piling up unbounded in the executor's completion queue. Fixed with real
   backpressure: at most `max_workers` futures in flight, next submitted only once a
   result is retrieved. Caps parent memory at O(max_workers x per-bucket result).
   (commit `c7c2fce`)

**General lesson for any future pool code here:** "the workers are bounded" is not
"the pipeline is bounded." The two unbounded reservoirs are (a) whatever the parent
accumulates between retrieving a result and freeing it, and (b) the executor's own
backlog of completed-but-unretrieved results when submission outruns consumption.
Neither is visible to `start_worker_watchdog`, which only watches worker RSS.

## B. The per-worker watchdog aborted whole experiments over a few MB -- **fixed**

`start_worker_watchdog` killed on the *first* sample over the *bare* per-worker
budget (`resolve_memory_budget_bytes // max_workers`), with zero tolerance or grace.
Because that budget is an even split of the aggregate, not a measured per-task
estimate, ordinary run-to-run variance put several workers a few percent over it at
the same moment -- and killing any one worker breaks the whole
`ProcessPoolExecutor`, so every sibling future then raises `BrokenProcessPool`.
Real result: **8 of 11 experiments aborted in a single batch run**, each killed
worker only 0-4% over budget. Fixed with `tolerance_fraction` (default 20% headroom)
and `grace_polls` (default 3 consecutive over-threshold polls before a kill) -- genuine
runaway growth keeps climbing across polls and is still caught; a worker merely
hovering near its fair share is not. (commit `a182e6c`)

## C. A single killed worker aborted the whole task list -- **fixed**

Even with (B) killing only genuine overages, one legitimately-killed worker still
took the entire stage down via `BrokenProcessPool`. `run_tasks_parallel` now catches
that, keeps already-completed results, and retries the still-pending tasks in a fresh
pool at half the worker count (halving on each repeated break, floor 1 = full
aggregate budget for one task). This makes the stage resilient regardless of how
wrong the per-item estimate is, short of one task alone exceeding the aggregate
budget. (commit `592d8b3`)

## D. The undercount was a ~27x memory *balloon* in materialize(), not a mis-tuned constant -- **fixed (2.16.0, commit `a8aa5c3`)**

Finding 4 estimated the per-task undercount at ~4-6x. Profiling the real run
(scratch harness, subprocess peak via `resource.getrusage`) showed it was far
worse *and* had a single concrete cause -- not the constant. A nominally-0.5GB
preprocess task peaked at **16-44GB**, and step-by-step breakdown put the entire
balloon inside `materialize()` (clean_NaN, transforms, core_skeleton added ~0):
`partition_read._load_ragged_selection` -> `ragged_store.read_ragged_parquet`
read *every* selected read's ragged arrays into one pandas frame before
`materialize_ragged` compacted it to dense. Those ragged list-columns are stored
as `int64`/`float64` (8 bytes for sequence/quality values that are 0-93) and held
as pandas object arrays, so **14,330 reads occupied ~13GB -- ~27x the 0.5GB dense
output**. That single-task peak alone exceeded the 6.4GB/worker budget, which is
the actual root cause of the whole batch-OOM saga (A/B/C were real and needed
fixing too, but this is why even one worker blew the budget).

Fixed by streaming the ragged->dense scatter: `materialize_ragged`'s
allocate/scatter/assemble logic was refactored into shared helpers, and a new
`materialize_ragged_streaming` preallocates the dense grid once (from `obs`, which
fixes shape/order/reference-lengths) and scatters ragged frames **chunk-by-chunk**,
freeing each before the next; `_load_ragged_selection` feeds it one parquet shard
at a time. Peak becomes ~one shard's frame + the dense output. Verified
byte-identical (X + every layer, dtypes, obs/var order) to the whole-frame path on
4 real 250822 tasks and via unit tests; **measured peak on the biggest real task
fell 16.3GB -> 4.4GB**, now under the per-worker budget so full parallelism fits
without invoking the (C) retry ladder.

**Still open (follow-up):** the 4.4GB residual is floored by `read_ragged_parquet`
reading each *whole* shard (~4200 reads) before filtering to the task's reads --
not chunk-proportional. pyarrow row-group/predicate filtering + storing the ragged
arrays as `int8`/`float32` (another ~4-8x) would push this lower and make the peak
proportional to the task chunk again, at which point `BYTES_PER_WORKING_POSITION`
could be recalibrated to an accurate small multiplier. Not needed for correctness
or to clear the OOM -- the streaming fix already does both.

## E. **NEW, not previously flagged:** nested, non-memory-guarded plotting pools

The reduce/plot phase of both HMM and spatial runs pools that bypass *all* of the
memory machinery above. `tools/partitioned_hmm.py::_plot_feature_clustermaps` (and
the equivalent spatial `_plot_read_metric_clustermaps`) runs in the parent after
`run_tasks_parallel`, loops references serially, and for each reference:

1. `materialize()`s a full adata for the entire reference-core -- **every read for
   that reference at once**, with no `reads_per_chunk` bound (unlike the bounded
   tasks). For a high-depth reference this is a large single parent-side allocation.
2. calls `combined_hmm_raw_clustermap(adata, ..., n_jobs=cfg.threads)`, which spawns
   its **own** `ProcessPoolExecutor(max_workers=cfg.threads)` (plotting.py:1372,
   2053) and pickles that whole adata to each of `cfg.threads` workers.

This second pool goes through none of `resolve_max_workers` (no budget-based
throttle), `start_worker_watchdog` (no kill protection), or `run_tasks_parallel`'s
retry ladder. On any machine where the preprocess tasks already bind memory, this
plotting pool will bind it too -- and unlike the task pools, it has no safety net, so
it fails the hard way (`BrokenProcessPool` straight out, or OOM). It did not surface
as the proximate failure in this batch only because preprocess (stage B/C) aborted
first. Recommended fix: route these plotting pools through `run_tasks_parallel` (or
at minimum `resolve_max_workers` + `start_worker_watchdog`), and chunk the
per-reference `materialize` in step 1 the same way the tasks do. Filed as a
follow-up to task #66.

---

# Performance logging design (per-command perf log next to the stage log)

**Motivation.** Every fix above was found by hand-running `ps`/`top`/`vm_stat`
loops beside the run. That monitoring should be a first-class, always-on artifact:
one perf log per command, sitting next to the existing stage log, capturing worker
counts and memory over time. It answers "how many workers did this command use and
how much memory did it need" without re-instrumenting by hand, and (because it's
per-stage-per-experiment) rolls up across a batch.

## File & format

Mirror `logging_utils.setup_stage_logging`, which already writes
`<stage_dir>/logs/<YYMMDD>_<HHMMSS>_log.log`. Add a sibling with the **same
timestamp stem** so the pair is obviously linked:

```
<stage_dir>/logs/260719_101151_log.log     # existing human log
<stage_dir>/logs/260719_101151_perf.jsonl  # new: one JSON object per line
```

JSONL, not free text: it stays greppable/tailable by a human but is trivially
aggregated across all stages/experiments in a batch (peak RSS, worker counts, wall
time) by a few lines of pandas. Gate with a new config `emit_perf_log` (default
True, mirrors `emit_log_file`); sampling cadence `perf_log_sample_interval_seconds`
(default = the watchdog's `DEFAULT_WORKER_POLL_INTERVAL_SECONDS`, 2.0).

## Event schema (one object per line)

- `pool_start`: `stage`, `worker_pool_id`, `n_tasks`, `max_workers`, and the *inputs*
  to the decision -- `threads`, `n_items`, `aggregate_budget_gb`,
  `target_task_memory_mb`, `by_memory_workers` -- so a reader sees *why* the worker
  count was what it was (this is exactly the mismatch that caused D).
- `sample` (every poll): `worker_pool_id`, `n_live_workers`, `parent_rss_gb`,
  `workers_rss_gb`, `tree_rss_gb`, `system_used_gb`, `system_available_gb`,
  `per_worker_over_budget` count. This is the `ps`/`vm_stat` loop, institutionalized.
- `pool_retry`: `worker_pool_id`, `reason` ("broken_pool"), `n_pending`,
  `new_max_workers` -- surfaces the (C) halving ladder.
- `pool_end`: `worker_pool_id`, `wall_seconds`, `peak_tree_rss_gb`, `n_retries`,
  `final_max_workers`.
- `stage_summary` (one per command, at stage end): `stage`, `wall_seconds`,
  `peak_tree_rss_gb`, `max_workers_used`, `n_pools`, `n_retries_total`.

## Where to hook it (single chokepoint, near-zero signature churn)

Two functions already sit at exactly the right places and already compute nearly all
of these numbers:

1. **`memory_guard.run_tasks_parallel`** already knows `max_workers`, `n_tasks`, the
   budget, and owns the retry ladder + pool lifecycle -> emits `pool_start` /
   `pool_retry` / `pool_end`. `_map_references_parallel` (raw) gets the same hooks.
2. **`memory_guard.start_worker_watchdog`** already polls every worker's RSS every
   `poll_interval` -- it currently only logs when *over* budget. Extend its poll loop
   to also emit a `sample` line every poll (add parent RSS + `psutil.virtual_memory()`
   for system used/available; it already has per-worker RSS and the live-pid set).
   This is the whole sampling mechanism, for free, in the one thread already doing it.

**Threading the handle through without touching every call site:** a
`contextvars.ContextVar[PerfLogger | None]` (in a small new `smftools/perf_log.py`,
or on `memory_guard`), set once per stage by `setup_stage_logging` (or a sibling
`setup_stage_perf_logging`) and read by the two functions above. When it's `None`
(tests, library use, `emit_perf_log=False`), every emit is a no-op -- no signature
changes to `run_tasks_parallel` / `start_worker_watchdog` / any caller. The parent
process is the only writer (workers are sampled *by* the parent's watchdog thread),
so no cross-process log contention. `stage_summary` is emitted by the stage wrapper,
which already calls `setup_stage_logging`.

## Batch rollup (nice-to-have, falls out of JSONL for free)

Because each stage writes its own `*_perf.jsonl`, a ~20-line helper can walk a run's
`*/logs/*_perf.jsonl`, take `max(peak_tree_rss_gb)` and the worker-count histogram
per stage, and print a per-experiment / per-batch scaling table -- directly the
"how much memory did this batch need, at what parallelism" question, without any
live monitoring.

## Scope note

This is logging/observability only -- it does not change scheduling or memory
behavior. It makes the D/E problems *visible in-band* on every run (the perf log
would have shown the 9->20GB worker climb and the plotting-pool spike immediately),
which is the prerequisite for tuning `target_task_memory_mb` / recalibrating
`BYTES_PER_WORKING_POSITION` from real numbers rather than by hand.

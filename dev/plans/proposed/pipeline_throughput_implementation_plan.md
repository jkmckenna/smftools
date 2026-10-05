# Pipeline throughput (`THR`)

**Status:** proposed overall. Merged: `THR-01` (#632), `THR-02` (#633), `THR-06` (#637).
Parked: `THR-03` (`F60`). Shelved: `THR-04` (`F62`). Proposed: `THR-05`.

Motivated by `F53`-`F57` in `logs/pipeline_findings.md`, measured on a
14-experiment `experiment batch full` regeneration (conversion and deaminase,
9.7k to 1.68M primary reads). Read those first; this plan does not restate the
measurements.

## Status

| item | status | evidence |
|---|---|---|
| `THR-01` alignment rescue: threaded BGZF, skip the rewrite when nothing is rescued | merged (#632) | `dc1f45c` on `fix/alignment-rescue-bam-threads`; `test_rescue_threaded_output_matches_unthreaded`; `F53` follow-up |
| `THR-02` latent: skip UMAP (not the unit) below UMAP's own minimum; no crash on tiny units | merged (#633) | `5c890ba` on `fix/latent-unit-read-floor`; `test_latent_unit_too_small_for_umap_keeps_pca_and_nmf` |
| `THR-03` duplicate detection: size-derived group memory estimate, largest-first dispatch | parked | since `DSA-06` (`F60`): 0 watchdog kills, 0 retries, dedup pools <1 min on four post-merge runs; estimate still 2-3x low -- revisit only if a large batch shows kills |
| `THR-04` latent: fit units in a worker pool | shelved | `F62`: CP must stay on the GPU (pool workers crash on MPS; CPU CP differs by up to 35%), and post-dedup units are few and mostly CP-eligible |
| `THR-05` `experiment batch`: run experiments concurrently under the memory envelope | proposed | -- |
| `THR-06` raw extraction: contiguous buckets read by virtual-offset range, not a full-contig scan per bucket | merged (#637) | `7f33fbb` on `fix/raw-contiguous-shards`; `test_scan_ranges_extract_exactly_the_full_reference`; `F59` |

Ordered by value per unit of risk. `THR-01` and `THR-02` are small and
independent; `THR-03` should land before `THR-05`, because concurrent
experiments multiply the cost of an under-estimated pool.

## Problem

On one batch, 632 of 981 min of stage wall time ran with no worker pool alive
(`F57`). Most of that is external or already threaded (alignment, sort, index),
but three phases are single-threaded Python or collapse to one worker:

- **Alignment rescue** rewrites the whole BAM on one core to flip a flag on
  0.1-0.8% of reads (`F53`). Over 88 min on the largest run.
- **Duplicate detection** sizes its group pool from a flat 512 MB per task. The
  watchdog then kills 22-46 GiB workers and the pool shrinks to one worker
  (`F54`). 135 min wall at ~6% pool efficiency on a 620k-read run.
- **Latent** fits units one at a time in the main process (`F55`), and crashes
  outright when a unit is too small for UMAP's spectral init (`F56`).

## Items

### `THR-01` Alignment rescue: threaded BGZF, skip the no-op rewrite

`informatics/alignment_rescue.py`, pass 2.

- Open both `AlignmentFile`s with `threads=` from the existing `threads`
  argument (already forwarded to re-indexing).
- When `promotions` is empty, do not rewrite: return the input BAM unchanged
  (or hard-link/copy it to `output_path` if callers require the path), and
  record that the rewrite was skipped in the summary.
- Log pass 2 start and end, so the step stops showing up as silence.

**Acceptance.** Output BAM records identical to the current implementation
(compare decoded records, not bytes; BGZF block boundaries may differ) on a
fixture with promotions and one without. Wall-clock for pass 2 on a real run
recorded in a finding, before and after.

**Not in scope.** Replacing the per-record Python loop. Revisit only if `F53`'s
follow-up shows it is the floor once compression is parallel.

### `THR-02` Latent: skip UMAP, not the unit, below UMAP's minimum

`tools/partitioned_latent.py`.

**As implemented (revised from the first draft).** The first draft proposed a
parameter-derived whole-unit floor. Measuring first showed that was the wrong
shape: with umap 0.5.12 the failure is exactly N = `n_components` + 1 = 3 fit
reads; N >= 4 succeeds, including disconnected graphs with 1-4 point
components and all-identical rows, and `n_neighbors` is already clamped to
N - 1. PCA and NMF already clamp their component counts. So the fix gates
only UMAP (and its Leiden clustering) on `_UMAP_MIN_FIT_READS` =
`n_components` + 2 and logs a warning; PCA/NMF are kept. Downstream already
tolerated a unit without UMAP (availability is per unit, and
`latent_min_reads: 2` could reach that state before). `latent_min_reads` is
unchanged.

**Acceptance.** Met: a 3-read unit test reproduces the production
`k >= N` error before the fix and keeps PCA/NMF without UMAP after it. Existing
latent tests pass. Not yet verified by a real-data latent rerun on the new
code.

### `THR-03` Duplicate detection: size-derived group estimate, largest-first dispatch

`preprocessing/partitioned_executor.py` (group pool), `memory_guard.py`.

- Add a `duplicate_detection_group_peak` estimator: per-group bytes from
  `len(core_obs)` and `load_end - load_start`, using the same per-cell factors
  `duplicate_detection_chunk_peak` already uses for chunks, plus any `DSA`
  anchor-window overhead. Pass a per-task estimate rather than one
  `per_item_memory_mb` for the whole pool, if `run_tasks_parallel` can admit
  tasks by individual size; otherwise use the largest group's estimate.
- Dispatch groups largest-first, so the longest task starts at t=0 instead of
  last.
- Calibrate the per-cell factor against `F54`'s measured peaks (22.5 GiB and
  45.6 GiB single workers) and record the calibration as a finding.

**Acceptance.** Union-find merge is order-independent (the code already
asserts this), so duplicate flags must be identical with and without
reordering. Verified on a real run: no `broken_pool` retry, recorded pool
efficiency and wall time against `F54`'s baseline.

**Coordinate with `DSA`.** `DSA-05` (real-data qualification) is open on the
same code. Record per-group peak RSS there too, so the estimator and the
anchor-window cost are measured once, not twice.

### `THR-04` Latent: fit units in a worker pool

`tools/partitioned_latent.py`.

- Dispatch units through `run_tasks_parallel` with a latent-specific estimator.
  Keep each unit's `random_state` fixed, so results are reproducible
  regardless of pool size or completion order.
- Set BLAS threads per worker so the pool does not oversubscribe cores.
- Collect results in unit order before publishing, so generation contents do
  not depend on scheduling.

**Acceptance.** Latent outputs identical (embeddings within floating-point
tolerance, cluster labels exactly) between a 1-worker and an N-worker run of
the same fixture. Wall time recorded on a real run against `F55`.

**Open question.** Whether a pooled run can stay bit-identical to today's
sequential run, or only to itself across pool sizes. Decide before
implementing; it determines whether existing latent generations must be
regenerated.

### `THR-05` `experiment batch`: concurrent experiments

`cli` batch command.

- Add `--jobs N` (default 1, today's behaviour). Admit the next experiment
  only when the resource envelope's available memory exceeds that
  experiment's predicted stage peak, so concurrency never relies on the
  watchdog to fit.
- Keep per-experiment logs separate; the batch summary records each
  experiment's outcome as today.

**Acceptance.** `--jobs 1` behaviour and summary unchanged. With `--jobs 2`,
no watchdog kill on a batch that completes cleanly at `--jobs 1`.

**May stay proposed.** `F57`'s CPU utilisation is not yet measured; if
`THR-01`, `THR-03` and `THR-04` close most of the serial time, this may not be
worth its complexity. `BCS-09` already notes that no batch orchestrator exists
to enforce scheduling within.

### `THR-06` Raw extraction: one scan per reference, not one per bucket

`cli/raw_adata.py`, `informatics/bam_functions.py`. Motivated by `F59`.

Raw extraction split each reference's reads round-robin into buckets of
`raw_bucket_max_reads`, and every bucket task streamed the *whole* reference
through `samtools view` to find its own reads. The docstring called that scan
cheap; on a 740k-read reference it was 97% of every task.

**As implemented.**

- The existing per-reference pre-scan (`_read_ids_and_offsets_for_reference`)
  now runs threaded and records each primary read's BGZF virtual offset,
  taken with `tell()` before the record.
- `_contiguous_buckets` splits those reads into the same number of buckets,
  but as contiguous runs in BAM order, sizes differing by at most one. Each
  bucket carries `(start_offset, end_offset)`. The split is by count, so the
  balance the round-robin split existed for -- position windows were badly
  imbalanced on amplicons with pile-ups at primer sites -- is unchanged.
- `extract_read_relative_base_identities(scan_range=...)` seeks to the
  bucket's start with pysam and stops at its end (or the reference's end).
  The read-name filter still applies.
- The pysam path now skips primary records stored without SEQ, as the
  samtools path always has. Real BAMs carry them (28 in one 4,000-read
  bucket); without the skip the pysam path raised on the first one. This was
  latent in the existing `python` backend too.

**Acceptance.** Met. On real data (a 740k-read reference, 185 buckets),
records from the new path are identical, field for field, to the samtools
full-scan path for the first, middle and last buckets: 3.4s against 77.7s per
bucket (22-23x; 165x for the last). Planning the whole reference took 4.3s
threaded. Unit tests cover range extraction against a full-reference
extraction on a multi-block BAM with pile-ups, secondary/supplementary and
SEQ-less records, a following contig and an unmapped tail, with planning
threads on and off.

**Raw algorithm version not bumped.** Per-read records are identical; only
which reads share a bucket changes, and bucket completion order already made
shard layout run-dependent. Pending regenerations are forced by `F58`'s
bump to `"4"` regardless.

## Not covered

- The raw read-feature scan and stage-boundary h5ad reads/writes (`F57`).
  Real but small; revisit after the items above.
- The external aligner's throughput. It is given `-t <threads>`; its
  utilisation has not been measured.

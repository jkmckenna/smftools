# Pipeline throughput (`THR`)

**Status:** proposed. Plan drafted on `docs/pipeline-throughput-plan`, cut from
`968383c`. No implementation branch yet; each item gets its own
`fix/<description>` branch from `main`.

Motivated by `F53`-`F57` in `logs/pipeline_findings.md`, measured on a
14-experiment `experiment batch full` regeneration (conversion and deaminase,
9.7k to 1.68M primary reads). Read those first; this plan does not restate the
measurements.

## Status

| item | status | evidence |
|---|---|---|
| `THR-01` alignment rescue: threaded BGZF, skip the rewrite when nothing is rescued | proposed | -- |
| `THR-02` latent: derive the per-unit read floor from method parameters; no crash on tiny units | proposed | -- |
| `THR-03` duplicate detection: size-derived group memory estimate, largest-first dispatch | proposed | -- |
| `THR-04` latent: fit units in a worker pool | proposed | -- |
| `THR-05` `experiment batch`: run experiments concurrently under the memory envelope | proposed | -- |

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

### `THR-02` Latent: a parameter-derived unit floor

`tools/partitioned_latent.py`, `config/experiment_config.py`.

- Compute the effective minimum read count per unit from the methods that will
  run on it: UMAP needs N > `n_neighbors` and N > `n_components` + 1 for
  spectral init; PCA/NMF need N > `n_components`. Skip, with the existing
  warning, any unit below `max(latent_min_reads, derived_floor)`.
- Keep `latent_min_reads` as a user floor; do not silently raise the default.

**Acceptance.** A unit test with N between the old floor (3) and `n_neighbors`
that crashes today and is skipped with a warning afterwards. A full-latent
fixture result unchanged for units above the floor.

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

## Not covered

- The raw read-feature scan and stage-boundary h5ad reads/writes (`F57`).
  Real but small; revisit after the items above.
- The external aligner's throughput. It is given `-t <threads>`; its
  utilisation has not been measured.

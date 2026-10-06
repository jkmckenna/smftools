# Materialize read cost (`MRC`)

**Status:** in progress. `MRC-01`, `MRC-02`, `MRC-04` merged; `MRC-03` (rescoped) implemented on
`fix/mrc-03-overlay-owning-stores`, on top of the `F71` fix, not merged; `MRC-05` proposed.

## Problem (`F70`)

`smftools.informatics.partition_read.materialize` is the one read path behind
ML partition reads (`MLX-06`, `MLX-09`, `MLX-11`), project materialization,
latent clustermaps and the plotting stages. On a real deaminase experiment,
one call for 800 reads that all sit in **one** preprocess partition costs
~16 s for the preprocess stage and ~17 s for the HMM stage. Profiled:

| cost | per call | cause |
|---|---|---|
| partition scan | ~13 s | `_load_preprocess_x_selection` finds the reads' partition in the derived read index, then still iterates **every** candidate partition of the reference in catalog order; a partition the index did not name falls into the `read_ids=` branch of `read_zarr_subset`, which opens it to look for the reads. ~40 opens to find 1. |
| spine load | ~2.6 s | `load_spine` re-reads the whole `spine.h5ad` on every call, also for the spine the previous call loaded. |
| partition open | ~0.2-0.3 s per open | `read_zarr_subset` reads each group's full obs (~60 columns) and var (~55 columns) through anndata before projecting rows and positions. |
| unneeded `X` | one more scan | Through an HMM-stage spine, a call asking only for derived layers (HMM features) still loads preprocess `X` through the fast path, then overlays the requested layers. |

Batched ML and latent reads pay this once per block, so it bounds every
whole-dataset read: ~30 molecules/s per process before `MLX-11`.

## Work items

| item | status | change | expected effect |
|---|---|---|---|
| `MRC-01` index-directed partition reads | merged | open only the partitions the read index names | ~16 s -> under 1 s + spine load, per call |
| `MRC-02` spine cache | merged | reuse a loaded spine within a process | -2.6 s per call after the first |
| `MRC-03` overlay opens only owning stores (rescoped) | implemented, not merged | skip a stage's store when its catalog lists none of the requested layers | HMM call 0.95 s -> 0.59 s |
| `MRC-04` skip `X` for derived-only requests | merged | do not load preprocess `X` when only derived layers are asked for | roughly halves HMM-stage calls |
| `MRC-05` qualification | proposed | before/after on real stores, end to end | -- |

Each item is behaviour-preserving: the same molecules, values, layers and
order come back. Each PR carries an equivalence test (result of the changed
path equals the result of the existing path on the same fixture) and a cost
test (count of partition opens, spine loads or `X` reads).

### `MRC-01` — index-directed partition reads

`_load_preprocess_x_selection`: when the derived read index resolved the
selection (`indexed_by_path` non-empty), visit only those `group_path`s and
skip every other candidate; keep the `read_ids=` scan only when there is no
read index (older runs). Iterate `indexed_by_path` rather than the catalog so
catalog order cannot matter. If the index covers the selection across more
than one partition, return `None` as today (the single-shard rule is
unchanged).

Tests: a fixture with many barcode partitions of one reference, the wanted
reads in the last one; `read_zarr_subset` is called once (was once per
partition), and the result equals the pre-change result. A run without a read
index still finds the reads by scanning.

**As implemented.** When `indexed_by_path` is non-empty, the candidate loop
keeps only the partitions it names; otherwise unchanged. Tests:
`test_fast_path_opens_only_the_indexed_partition` (six barcode partitions,
the read in the last; only that partition is opened -- once for `X`, once for
the layer overlay; fails on the old loop), `test_fast_path_still_scans_without_a_read_index`.
Real store, one call for 800 reads of one partition: preprocess 11.1 s / 39
partition opens -> 2.7 s / 1; HMM 11.9 s / 41 -> 3.6 s / 3; `X` and layer sums
identical. What remains is mostly the spine reload (`MRC-02`).

### `MRC-02` — spine cache

`_resolve_spine` loads through a small process-local cache keyed on the
resolved path, size and mtime (a re-published spine is a new key), bounded
(LRU, a few entries) so memory stays flat across experiments. Callers get the
cached object; `materialize` must not mutate the spine (audit: it reads
`uns`/`obs` only -- confirm, and copy where it does not). An opt-out
(`SMFTOOLS_SPINE_CACHE=0`) for debugging.

Tests: two calls on one spine load it once; touching the file reloads it; the
LRU evicts; results equal uncached results.

**As implemented.** `_resolve_spine` loads through `_cached_spine` (key:
resolved path, size, mtime_ns; LRU of 4; lock-guarded; `SMFTOOLS_SPINE_CACHE=0`
opts out; `clear_spine_cache()`). `load_spine` itself stays uncached: other
modules load spines to modify and rewrite them. Audit: `materialize` never
writes to the spine; the selection is a copy (`obs.loc[mask]`); the two places
that handed `spine.uns` values to a result by reference (the per-partition
path and the ragged path) now deep-copy them -- the test that edits a result's
`uns` caught the ragged one. Tests: `test_spine_cache.py` (one load for repeat
calls, reload after rewrite, LRU bound, cached == uncached, result edits do
not reach the cached spine). Real store, second call on the same spine:
preprocess 2.54 s -> 0.46 s, HMM 3.39 s -> 1.23 s; values identical.

### `MRC-03` — overlay opens only stores that wrote a requested layer (rescoped)

**Rescoped from column-projected opens.** Measured after `MRC-01`/`02`/`04`,
reading obs/var dataframes costs ~0.01 s (preprocess) to ~0.14 s (HMM) per
call -- under 10% -- so projecting columns was not worth a PR and an opt-in
API. The eager full-partition reads in the original profile came from the
`read_ids=` scan that `MRC-01` removed; the lazy path works on these stores.
The same profile showed the real waste: `_overlay_preprocess_layers` loops over
both derived stages' stores for every request, so asking for HMM layers also
opened the matching preprocess partitions, which cannot hold them.

**As implemented.** Each stage's catalog `layers` column (already read by the
`F71` fix) says what it wrote; a stage listing none of the requested layers is
skipped. Catalogs without a `layers` column are read as before. Test:
`test_overlay_skips_stages_that_wrote_none_of_the_requested_layers` (a second
stage over the same partitions listing another layer: 3 opens before, 1
after). Real HMM call, 800 reads, warm spine: 2 opens -> 1, 0.95 s -> 0.59 s;
layer sums identical. Preprocess calls unchanged.

### `MRC-04` — skip `X` for derived-only requests

When the requested layers are all derived (overlaid from another stage's
store) and the caller does not need `X`, build the result from the selection
and the overlays alone instead of first loading preprocess `X`. Needs an
explicit flag (`x=False` or similar) so existing callers keep `X`; the ML
reader passes it for channels that read only derived layers.

Tests: derived-only results equal the derived layers of a full read; no `X`
read happens (patched loader count).

**As implemented.** `materialize(..., x=False)`: with an explicit layers list
that is entirely derived, the result is built from the selection's rows and
the window's positions (`_rows_and_positions`, with `position_in_<reference>`
marked as the `X` path marks it) and the same overlays run; `X` is `None`.
Anything else with `x=False` raises. The ML reader passes `x=False` for a
stage whose channels read no `X` layer, and falls back to reading `X` if the
stage's layers are not all derived. On a real HMM stage the two paths give the
same rows, positions, layer values and shared var columns; the `X` path also
carries per-partition preprocess summary columns (`*_partial`) that
derived-only readers do not use. Warm call, 800 reads: 1.21 s -> 0.91 s
(~25%, not the ~50% estimated -- the overlay dominates). Tests:
`test_derived_only_read_matches_full_read_without_reading_x`,
`test_derived_only_read_refuses_other_requests`.

### `MRC-05` — qualification

On a real project: per-call cost before/after each item for one partition's
reads (preprocess and HMM stages), a whole-dataset read with 1 and 8 workers,
and an ML fold train pass; values identical to a pre-`MRC` read of the same
rows.

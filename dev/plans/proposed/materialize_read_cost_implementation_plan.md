# Materialize read cost (`MRC`)

**Status:** proposed. Nothing implemented. One PR per item, in order.

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
| `MRC-01` index-directed partition reads | proposed | open only the partitions the read index names | ~16 s -> under 1 s + spine load, per call |
| `MRC-02` spine cache | proposed | reuse a loaded spine within a process | -2.6 s per call after the first |
| `MRC-03` column-projected partition opens | proposed | read only the obs/var columns a subset needs | lower cost per open |
| `MRC-04` skip `X` for derived-only requests | proposed | do not load preprocess `X` when only derived layers are asked for | roughly halves HMM-stage calls |
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

### `MRC-02` — spine cache

`_resolve_spine` loads through a small process-local cache keyed on the
resolved path, size and mtime (a re-published spine is a new key), bounded
(LRU, a few entries) so memory stays flat across experiments. Callers get the
cached object; `materialize` must not mutate the spine (audit: it reads
`uns`/`obs` only -- confirm, and copy where it does not). An opt-out
(`SMFTOOLS_SPINE_CACHE=0`) for debugging.

Tests: two calls on one spine load it once; touching the file reloads it; the
LRU evicts; results equal uncached results.

### `MRC-03` — column-projected partition opens

`read_zarr_subset` reads obs/var through anndata's full-dataframe readers
before slicing. Read only the obs columns the caller keeps (identity columns
plus any requested) and the var columns needed for position selection and
design (`<reference>_*_site`, `position_in_*`), lazily where anndata allows.

Tests: projected and unprojected reads give identical `X`, layers, obs names
and kept columns; a wide-obs fixture shows fewer column reads. Decide in the
PR whether callers that rely on every obs column (plotting) opt in to the full
read.

### `MRC-04` — skip `X` for derived-only requests

When the requested layers are all derived (overlaid from another stage's
store) and the caller does not need `X`, build the result from the selection
and the overlays alone instead of first loading preprocess `X`. Needs an
explicit flag (`x=False` or similar) so existing callers keep `X`; the ML
reader passes it for channels that read only derived layers.

Tests: derived-only results equal the derived layers of a full read; no `X`
read happens (patched loader count).

### `MRC-05` — qualification

On a real project: per-call cost before/after each item for one partition's
reads (preprocess and HMM stages), a whole-dataset read with 1 and 8 workers,
and an ML fold train pass; values identical to a pre-`MRC` read of the same
rows.

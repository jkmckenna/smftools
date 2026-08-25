# smftools output re-architecture: thin molecule-index AnnData over distributed storage

> Working plan doc (mirrors the approved planning-session plan). Tracks the multi-branch
> re-architecture of smftools storage/CLI. Branches 1.0.0 and 1.1.0 are built; the
> **current active work is v2.0.0 — the `raw`/`load` split** (section below).

## Context

Today every smftools stage centers on one monolithic AnnData written as `.h5ad.gz`
(uncompressed HDF5). `smftools load` bakes everything into it — per-read scalar
metadata in `obs`, and four heavy `n_reads × n_positions` matrices in `layers`
(`sequence_integer_encoding`, `mismatch_integer_encoding`, `base_quality_scores`,
`read_span_mask`). Two structural problems block scaling to millions of reads and
multi-experiment projects:

1. **Every downstream stage fully materializes the whole object** via
   `safe_read_h5ad(source_path)` (`cli/spatial_adata.py`, `hmm_adata.py`,
   `variant_adata.py`, `latent_adata.py`, `chimeric_adata.py`). Peak RAM is
   O(entire dataset) in every stage.
2. **Cross-experiment combination is an eager merge**: `concatenate_h5ads`
   (`readwrite.py`) reads all experiments fully into memory, then `ad.concat`.

Also: all references are padded to one `max_reference_length` despite each
reference having its own valid-position set.

**Target model.** The per-experiment AnnData becomes a **thin "molecule-index
spine"**: one `obs` row per molecule (read), carrying only identity + *pointers*
linking each molecule to its data distributed across the aligned BAM, zarr
partitions, and parquet sidecars. Heavy matrices and scalar metadata live outside
the AnnData and are materialized on demand. A **project-level light AnnData** then
indexes many experiments and loads their data as needed. Stay in the anndata
ecosystem: zarr + dask + a lightweight DuckDB/parquet catalog.

## Branch & version strategy

Stacked, dependent branches off `0.4.5-layer-audit`. Version in
`src/smftools/_version.py` (hatch dynamic).

- **Branch 1 — `1.0.0-partitioned-store`** (DONE): partitioned zarr, sidecar
  generalization, thin spine, migration.
- **Branch 2 — `1.1.0-lazy-stage-reads`** (materialize DONE): materialize API;
  convert CLI stages to partition-aware reads.
- **Branch 3 — `1.2.0-project-catalog`**: project catalog + incremental
  experiment registration. (Deferred behind 2.0.0.)
- **Branch 4 — `2.0.0-load-split`** (off branch 2): add `smftools raw` (ragged
  read-relative source of truth); make dense an on-demand materialization +
  optional cache; `load` becomes the optional dense-cache builder; port stages to
  the unified accessor. Breaking CLI change → major bump. **Current active work.**

## v2.0.0 — `raw` (ragged source of truth) + on-demand densification (`load` = optional cache)

**Why.** Today `load` conflates BAM-level tool-heavy work (basecall / align / demux /
modkit) with dense-matrix construction, and forces a dense reference-grid matrix as a
**mandatory** intermediate every downstream stage depends on. Two changes: (1) split
the BAM work into `smftools raw`, which extracts per-read metrics in **read-alignment
coordinates** (ragged, indexed by each read's own aligned positions — NOT padded onto
a shared reference grid); (2) make the dense matrix an **on-demand materialization +
optional cache**. Read-level analyses run directly off the ragged artifacts;
position-wise analyses densify only the slice they need (from cache if built, else
from ragged); `load` becomes an **optional** command that pre-builds/persists the
full dense zarr store as a cache.

**`smftools raw`** (new; today's `load` sections 1–6 in `load_adata_core`): input
handling, FASTA, basecall (dorado), align/sort, UMI/barcode extraction, demux, BAM
QC, modkit extract. Then a **read-relative per-read extraction** — refactor
`bam_functions.extract_base_identities` (conversion) and the modkit-TSV parse
(direct) to emit per-read arrays in read coords, NOT scattered to the reference grid.
Output = **parquet per-read sidecar(s)** (Arrow variable-length `list` columns) keyed
by `read_id`: `reference_start`, `cigar`, `aligned_length`, strand, reference, barcode
+ ragged arrays (mod signal / base identity, quality, mismatch). Read-span is scalars
(`reference_start` + `aligned_length`). Shardable: one parquet per worker/BAM. Emits
BAM + accessories + sidecars + manifest (reuse `cli/helpers.py::ArtifactPaths` +
`informatics/sidecar_manifest.py`). **No dense matrix, no zarr.**

**Unified accessor — `materialize(selection, ...)`** (extend
`informatics/partition_read.py`). Resolves a selection from EITHER: the **dense zarr
cache** (fast read; existing 1.1.0 path) when it exists, OR **densify-from-ragged**:
read the selection's rows from ragged parquet and walk each read's CIGAR to scatter
its arrays onto the reference grid → dense AnnData slice (placement is the second half
of today's `extract_base_identities`, now a standalone helper shared with `load`).
Same return type either way. Read-level-only consumers bypass it and read parquet.

**`smftools load`** (now OPTIONAL, any-source): materialize + **persist** the full
dense zarr store from `raw`'s parquet — or any prior stage's output — writing
per-`Reference_strand` partitions (configurable `partition_by`; default
`[Reference_strand]`, pooling barcodes; `[Reference_strand, Sample]` optional), obs
ordered by `Barcode`. **Balanced parallel write:** preassign each read a row offset
(group by reference, order by barcode); split reads into equal-count work-chunks for
the placement compute (decoupled from layout); each worker writes its placed rows into
the preallocated per-reference zarr partition at its offsets (work-chunks aligned to
zarr obs-chunks; ordered-merge fallback). Record per-(reference, barcode) row-ranges in
the catalog. Then build spine + catalog + obs. Reuse `informatics/partition_store.py`.

**Stage porting** — convert `cli/*_adata.py` + `cli/preprocess_adata.py` off
`safe_read_h5ad(full)`:
- **read-level steps** (`filter_reads_on_length_quality_mapping`,
  `calculate_read_modification_stats`, `filter_reads_on_modification_thresholds`) →
  read ragged parquet directly; no densification.
- **position-wise steps** (`calculate_coverage`, `append_base_context`,
  `binarize`/`clean_NaN`, `flag_duplicate_reads` Hamming, spatial autocorrelation, HMM,
  variant per-position frequencies, latent PCA/UMAP) → `materialize(selection)` per
  reference/window.

**Derived-data storage (stage outputs).** All per-read arrays share the one
per-reference, barcode-ordered **row layout** (row *i* = same read across every store):
- **per-read scalar (obs)** → per-stage **obs parquet sidecar** keyed by `read_id`.
- **ragged read-relative layer** → **parquet list columns** (the `raw` pattern; rare).
- **dense reference-grid layer** (binarized / nan-fill / `*_site_binary` / HMM masks +
  emissions) → **per-stage zarr store** (`preprocess.zarr`, `hmm.zarr`, `latent.zarr`),
  per-reference partitioned on the shared row layout.
- **obsm (reads × K)** (PCA/UMAP/NMF/spatial) → **2D zarr arrays** in the per-stage
  store, parallel to layers, same row layout.
- **var / varp** → per-reference `var` columns / zarr.
Each derived key is indexed `key → store` in the spine `uns` / manifest, so
`materialize(selection, layers=…, obsm=…, obs_cols=…)` assembles a full AnnData by
pulling each requested piece from its store and aligning by row.

**Filtering = masks, not deletion.** Read filtering (QC drops, dedup) writes boolean
**obs masks** (`passes_qc`, `is_duplicate`, …) + a filtered **spine view**; no physical
row removal. Every derived store stays aligned to the single full per-reference layout.

**Storage tiering (final):** BAM = reads + native MM/ML; **parquet sidecars = ragged
per-read metrics (`raw`; source of truth)**; **zarr = dense reference-grid matrices
(optional raw-derived cache built by `load`, + per-stage derived stores)**.

**Files.** New: `cli/raw_adata.py` + `raw` command in `cli_entry.py`; per-read parquet
writer/reader. Refactor: `informatics/bam_functions.py::extract_base_identities` →
read-relative extractor (`raw`) + CIGAR-placement helper (shared by `materialize` +
`load`); `informatics/modkit_extract_to_adata.py` similarly. Extend:
`informatics/partition_read.py::materialize` (densify-from-ragged). Modify:
`cli/load_adata.py`, `informatics/partition_store.py`, `cli/*_adata.py` +
`cli/preprocess_adata.py`, `cli_entry.py`, `cli/helpers.py`, `_version.py` → 2.0.0.

**Verification.** Unit: read-relative extraction + CIGAR placement == current
`extract_base_identities` reference-grid output (exact); parquet round-trip;
**materialize parity — densify-from-ragged == read-from-cache**; balanced-chunk layout
yields barcode-ordered partitions with correct per-(ref,barcode) row-ranges.
Equivalence: `raw` (+ optional `load`) then a stage == the 1.x monolithic pipeline on a
small real dataset; a read-level stage runs off ragged without any dense store.
Regression: `venvs/venv-0.4.0/bin/python -m pytest tests/unit -q`. Resource: equal read
counts/worker; stage peak RSS bounded by its densified slice.

### Current implementation checkpoint

- Raw parquet is partitioned by `Reference_strand` and genomic start bin, with a
  shard-level interval catalog and read-span pointers on the thin spine.
- `analysis_mode` and `load_cache_mode` are planned independently per reference;
  short references use full dense caches and genome references use haloed tiles or
  no cache.
- `materialize(..., start=, end=)` reads bounded raw or cache intervals and can
  stitch preprocessing-derived layers across task cores.
- Partitioned preprocessing dispatches by reference, core/halo, barcode, and
  memory-bounded read chunk. Empty cores are skipped.
- `preprocess_execution_mode: auto` selects partitioned preprocessing for any
  planned raw/load spine, including locus-only experiments. Legacy preprocessing
  remains available for monolithic H5AD inputs or explicit `legacy` mode.
- Local binarization/NaN layers and base context are written to task zarr stores.
  Coverage and per-read modification statistics are reduced into `var.parquet` and
  `obs.parquet`; filtering is represented by boolean masks rather than row deletion.
- Duplicate/Hamming clusters are reconciled across genomic cores. Remaining work
  includes bounded output ports for spatial, HMM, variant, latent, and chimeric stages.
- Each partitioned stage owns a `plots/` tree with stage-specific categories,
  `context.json` linking the source spine, and `catalog.parquet` indexing figures
  by plot type, reference, sample, and genomic core.

## anndata API decisions (from scverse tutorials)
- **materialize** uses `anndata.experimental.read_lazy(zarr_store)` (zarr-only; dask X/
  layers; obs/var eager xarray `Dataset2D`); `adata[mask].to_memory()`.
- **Project AnnData = `AnnCollection` over per-experiment spines** (obs-only → no
  var-join conflict). Data pulled per-reference (shared `var`). Incremental add = append.
- **zarr v3** default/auto-consolidates in newer anndata; on 0.12.9 pin via
  `settings.zarr_write_format = 3` (done in `safe_write_zarr`). `auto_shard_zarr_v3`
  (experimental) cuts file count for per-reference partitions.

## Environment note
- Test/build venv: `venvs/venv-0.4.0/bin/python` (anndata 0.12.9, zarr 3.1.5, dask,
  torch, pytest, ruff, xarray). Default `python` lacks torch.
- **Cannot adopt anndata 0.13** (requires numpy 2, conflicts with smftools/numba/captum
  `numpy<2`). Use 0.12.9 + `zarr_write_format=3` for v3 output.

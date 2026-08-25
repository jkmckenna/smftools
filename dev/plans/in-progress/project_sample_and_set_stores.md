# Design: project-level per-sample and set stores

> **Set store v2 is now implemented** (see [Proposed redesign: partition-native set
> store](#proposed-redesign-partition-native-set-store-v2) below, which documents the
> shipped design). The old `materialize_set`/`base.h5ad` concat cache is **removed**. A
> set is now a *query*, streamed one projected slice at a time via
> `set_store.iter_set_parts`; `catalog.project_adata` is the explicit, size-guarded
> "pool to one object" opt-in. The "Set store" section immediately below describes the
> superseded v1 design and is kept only for history.

**Status:** All 4 phases implemented. `src/smftools/project/sample_store.py` (wired
into `project add` via `cli/project_cmd.py`), `src/smftools/project/set_store.py`,
`src/smftools/project/sample_analysis.py`, `src/smftools/project/embedding_store.py`.

**CLI wiring:** `project materialize` now goes through `set_store.materialize_set`
(cached by resolved composition, `--force-recompute` to bypass) instead of calling
`catalog.project_adata` directly -- transparent, no behavior change for existing
callers beyond the new caching and the new flag. `project sample-store-list` is a new
read-only command over `sample_store.list_per_sample_partitions`. Not yet wired:
`sample_analysis.compute_periodicity`/`join_periodicity` (needs several new options,
and a decision on how `**kwargs`-style LS parameters map to flags) and
`embedding_store.fit_or_extend_embedding` (needs a decision on what actually gets
written/printed, since the result is several arrays + models, not one file the way
`project materialize` produces one h5ad).

Remaining work is the open questions below (GC of stale caches, definition-hash
canonicalization, reference-alias interaction, UMAP drift detection).

## Motivation

Two problems observed in practice (registering the `Nkg2a_DAFseq_merged` project's 11
legacy runs into a v2 project):

1. **Legacy monolithic experiments are expensive to re-read.** `materialize()`'s legacy
   path (`_materialize_legacy`) has no lazy/partial-read option -- every query against a
   legacy `.h5ad`/`.h5ad.gz` fully loads that file. A raw-stage materialize across 8
   legacy experiments (~2.7M reads) exhausted memory on a laptop. Repeating that full
   load on every analysis is wasteful when the underlying file never changes.
2. **Project-level analyses have two different shapes**, and neither has a home today:
   - **Per-sample analyses** (autocorrelation, periodicity over a defined read span) are
     computed independently per experiment and don't need a set definition at all.
   - **Set-level analyses** (PCA, KNN, Leiden clustering) need a set definition (which
     experiments, which canonical reference) and may need to grow as new experiments are
     registered, without redoing everything from scratch every time.

This document proposes two project-level stores -- a **per-sample store** and a **set
store** -- and how they relate to each other and to the existing registry/materialize
machinery.

## Goals

- Cache expensive-to-reread legacy data once, at registration time.
- Avoid ever duplicating data that's already cheaply reachable (modern partitioned
  experiments) -- point at it instead.
- Let per-sample analyses (autocorrelation, periodicity, ...) be computed once per
  experiment and reused by every set that includes it, not recomputed per set.
- Let set-level embeddings (PCA/UMAP) grow incrementally by default when a set's
  membership grows, with an explicit opt-in to a full recompute.
- Reuse existing conventions rather than inventing new ones: relative-path pointers,
  content-addressed invalidation (`reference_uid`-style hashing), and the
  overlay-catalog pattern `materialize()` already uses for preprocess/hmm derived
  layers.

## Non-goals

- This does not change how a single experiment's own pipeline stores data
  (`raw_outputs/`, `preprocess_adata_outputs/`, etc.) -- those are untouched.
- This does not attempt true streaming/out-of-core materialization for legacy files;
  the legacy cache is still one full read, just done once instead of repeatedly.
- This does not pick a specific embedding library API (scikit-learn PCA vs. custom, UMAP
  vs. some other backend) -- "persist a fitted model, support `.transform()`" is a
  requirement, not an implementation.

## Terminology

- **Per-sample store**: a project-level cache, one entry per
  `(experiment_id, Reference_strand, sample)`, holding either a pointer (modern
  experiments) or a full cached copy (legacy experiments) of that partition's reads,
  plus a catalog of project-computed per-sample analyses layered on top.
- **Set store**: a project-level cache of one set's base materialization (a specific
  `(set_name, canonical_reference, stage)` combination) plus any set-level derived
  embeddings/clusters.
- **Modern experiment**: registered from a directory with partitioned-store spines
  (`uns["is_spine"] = True`); supports lazy/dense-cache reads today.
- **Legacy experiment**: registered from a single monolithic `.h5ad`/`.h5ad.gz` file
  (see the legacy-adapter work); always a full eager read today.

## Per-sample store

### Partition key

`(experiment_id, Reference_strand, sample)` -- not sample alone.

Two reasons, both concrete in this project's own data:

- Barcodes collide across experiments (barcode `01` names a different biological
  sample in every run), so `experiment_id` has to be part of the key.
- `Reference_strand` has to be part of the key because per-sample analyses like
  autocorrelation are computed per locus, not pooled across a sample's unrelated
  amplicons -- and several registered experiments here already carry 4-8 references
  each. This also matches the granularity the existing task-based zarr stores already
  partition on at the run level (reference x barcode), just with `experiment_id`
  prepended for cross-run uniqueness, and lines up with how canonical-reference
  harmonization already resolves `(experiment, Reference_strand) -> canonical_reference`.

### Modern experiments: pointer, not copy

For an experiment registered from a partitioned-store directory, the per-sample store
entry is a relative-path pointer into that experiment's own task store (same convention
`_relative_registry_path`/`resolve_relative_path` already use), plus a project-local
overlay catalog for anything the *project* computes on top (autocorrelation,
periodicity, ...). This is the same pattern preprocess/hmm/spatial already use to layer
derived data onto a spine without duplicating the underlying reads -- the per-sample
store is that pattern one level up, spanning experiments instead of stages.

### Legacy experiments: cache at registration time

For an experiment registered from a legacy monolithic file, `project add` backfills the
per-sample store immediately: materialize the latest stage once (full eager read, same
as today), split it by `(Reference_strand, sample)`, and write each partition into the
project's own store (e.g. a per-partition zarr or parquet + `.npz`-style shard,
consistent with how the codebase already writes task partitions elsewhere). Every later
query -- per-sample analysis or set materialize -- reads from this cache, not from the
original legacy file again. The original file is still never mutated (same invariant as
the existing legacy adapter).

### What lives in a per-sample store partition

- A pointer or cached copy of that partition's molecules (see above).
- A catalog of project-computed per-sample analyses, each keyed by read_id, e.g.:
  `autocorrelation/<read_span_definition_hash>/...`, `periodicity/<read_span_definition_hash>/...`.
  Keying by a hash of the read-span definition (not just the analysis name) means
  changing how a read span is defined naturally produces a new cache entry instead of
  silently reusing a stale one, without needing manual versioning.
- Recompute trigger: only when the read-span/analysis definition changes -- never when
  set membership changes, since these are computed independent of any set.

## Set store

### Base materialization cache

Cached under something like `project_outputs/sets/<set_name>/<composition_hash>/base.h5ad`
(or a lighter columnar format), where `composition_hash` is a hash of the set's
*resolved* composition: which experiment/stage/spine paths and canonical reference
actually went into it. When a set's registry-resolved membership changes (a new
experiment registered, or the set's query now matches something new), the hash no
longer matches, the old cache is recognized as stale automatically, and a fresh
materialize + recompute happens -- no manually-tracked cache version needed. This
reuses the same content-addressing idea `reference_uid` already uses for reference
identity, just applied to set composition.

### Cross-linking to the per-sample store

The set store does not duplicate per-sample metrics. At set-materialize time, per-sample
catalog entries for every member `(experiment_id, Reference_strand, sample)` are joined
onto the set's base frame by read_id -- the same mechanism `materialize()` already uses
today to overlay preprocess/hmm derived layers and spatial `read_metrics` onto a single
experiment. A set growing by one experiment means one more join key, not a recompute of
anything per-sample.

### Embedding lifecycle

Persist the **fitted model**, not just its output coordinates -- PCA components, a
UMAP transformer, etc. -- as a first-class artifact of the set store
(`embeddings/<method>/model.<ext>` alongside `coords.npy`).

- **Default when a set grows**: transform the newly-added molecules through the
  persisted model (cheap, exact for PCA; approximate but supported for UMAP) and append
  their coordinates. Existing points' coordinates do not move.
- **Clustering (Leiden) does not have a clean incremental equivalent** -- a
  graph-community recompute over an expanded neighbor graph can shift boundaries for
  every point, not just new ones. Default for clustering specifically: assign new
  points a label via nearest-neighbor lookup against the fixed embedding; existing
  labels are left untouched.
- **Explicit recompute** (opt-in, not automatic): full refit from scratch across the
  set's current membership. Written as a new version
  (`embeddings/<method>/<refit_timestamp_or_hash>/...`) rather than overwriting the
  previous one, since analyses/figures built on stable coordinates shouldn't silently
  shift underneath them.

Tradeoff to carry forward: transform-only keeps continuity and is cheap, but the
embedding can quietly degrade in quality as the underlying space drifts further from
what the original fit captured. Transform-by-default is a reasonable default, not a
substitute for periodically refitting.

## Data flow

**Registration (`project add`)**

```text
project add <project_dir> <experiment_dir_or_file> [--stage ...]
  -> registry: record spine pointer(s) (existing behavior, unchanged)
  -> per-sample store backfill:
       modern experiment  -> write pointer + empty overlay catalog per (ref, sample)
       legacy experiment  -> materialize latest stage once, split + cache per (ref, sample)
```

**Set materialize (`project materialize --set ...`)**

```text
resolve set -> experiment list (existing behavior, unchanged)
compute composition_hash over resolved (experiment, stage, spine_path, canonical_reference)
if project_outputs/sets/<set_name>/<composition_hash>/base.h5ad exists:
    load cached base
else:
    materialize + concat each member experiment (existing behavior)
    join in per-sample store catalogs by read_id
    cache as base.h5ad under the new composition_hash
```

**Set-level analysis (PCA/UMAP/Leiden)**

```text
if embeddings/<method>/model exists and set only grew (no member removed/changed):
    transform new molecules through persisted model -> append coords
    assign new points to nearest existing cluster
else if --recompute:
    full refit across current membership -> new versioned embedding
```

## Actual layout (as implemented -- see note below on why it differs from the original proposal)

```text
<project_dir>/
├── registry.json
├── sets/                                  # existing: named set definitions
├── project_outputs/
│   ├── per_sample/
│   │   └── <experiment_id>/
│   │       └── <Reference_strand>/
│   │           └── <sample>/
│   │               ├── pointer.json        # modern: relative path into experiment store
│   │               ├── cache.h5ad          # legacy only: cached molecules
│   │               └── analyses/
│   │                   └── <analysis_name>/<definition_hash>/...
│   └── sets/
│       └── <set_name_or_canonical_reference>/
│           ├── <composition_hash>/          # one per resolved membership (Phase 3)
│           │   ├── base.h5ad
│           │   └── composition.json
│           └── embeddings/                  # NOT nested under composition_hash -- see below
│               └── <embedding_definition_hash>/
│                   ├── pca_model.pkl
│                   ├── umap_model.pkl
│                   ├── pca_space.npy
│                   ├── coords.npy
│                   ├── clusters.npy
│                   ├── obs_names.json
│                   ├── meta.json
│                   └── versions/<timestamp>/...   # only present after force_recompute
```

**Why embeddings live outside `<composition_hash>/`, unlike the original proposal
above this section**: the composition hash (Phase 3) is deliberately designed to
change on *any* resolved-membership change, including pure growth -- that's what
makes the base-materialization cache self-invalidating. But an embedding needs to
survive across exactly that kind of growth in order to be extendable at all (the
whole point of persisting the fitted model). Nesting it under `<composition_hash>/`
would mean every new experiment registration silently orphans the previous
embedding directory instead of extending it. So `embeddings/` is keyed only by the
embedding-defining parameters (feature choice, window, Leiden resolution, ...),
never by resolved membership -- discovered this while implementing Phase 4, after
the file layout above had already been proposed.

## Proposed redesign: partition-native set store (v2)

**Status: IMPLEMENTED.** Written after running `project materialize` against the real
11-experiment `Nkg2a_DAFseq_merged_v2` project surfaced the problem below; then built.
Supersedes the "Set store" section above. As shipped:
`set_store.iter_set_parts` (streamed, projected; `materialize_set`/`set_cache_dir`/
`base.h5ad` removed), `catalog.project_adata` (now consumes the stream + an ~8 GiB
size guardrail via `allow_large`/`max_bytes`), `embedding_store` (pools only the single
feature layer via `project_adata(layers=[...], allow_large=True)`), and the CLI
(`project materialize` gained `--layers`/`--allow-large`, dropped `--force-recompute`).
Per-experiment (not yet per-sample) partition granularity -- see the note under
"Proposed shape". Tests: `tests/unit/test_project_set_store.py` (iter/projection/lazy/
no-disk-writes), `test_project_catalog.py` (guardrail + layer projection),
`test_project_cli.py` (`--layers`/`--allow-large`, no `sets/` written).

### The problem with what shipped (Phase 3)

What actually shipped in `set_store.py` is *not* partition-native. `materialize_set`
wraps the pre-existing `catalog.project_adata`, which:

1. re-materializes each member experiment's full slice **from the registry spine**
   (`resolve_set_members` -> `spine_path`), **not** from the per-sample store, and
2. `ad.concat`s all members into **one monolithic AnnData**, cached as a single
   `base.h5ad`.

Three consequences, all observed on real data:

- **The per-sample store is bypassed entirely.** It's populated by `project add` but
  read only by `sample_analysis.py`. `project materialize` never touches it -- so the
  partitioned store and the set store are two parallel mechanisms that don't compose.
- **Peak memory is ~2x the pooled object.** All member parts stay resident in the
  `parts` list while `ad.concat` builds the result on top. Measured: ~56 GB RSS for
  the full 220k-read x 4690-position pool, with the machine down to ~3 GB free.
- **The pooled object is enormous and often unwritable.** `ad.concat(join="outer")`
  NaN-fills the union position axis, which **upcasts every int8 layer to float**
  (4-8x), and the pool carries **all ~15-25 layers** (a legacy HMM stage's
  accessibility/nucleosome/footprint layers) at full locus. Measured: the full-project
  pool exceeded **200 GB** uncompressed (`safe_write_h5ad` is uncompressed by design)
  and hit ENOSPC mid-write; even `--stage hmm` (dropping the one 188k-read raw-only
  experiment) still produced a **64 GB** monolith. The `var` `column-order` HDF5
  attribute also overflowed 64 KB from ~1200 per-experiment var columns (fixed
  separately by stripping var to the position axis, but the layer bloat remains).

The concat is a holdover: `project_adata` was the original project catalog, and
Phase 3 wrapped it with a cache instead of rethinking it. It reintroduces exactly the
monolithic-AnnData problem the partitioned architecture was built to escape.

### Two independent levers

1. **Layer/window projection (cheap, do this regardless).** Analyses need *one* layer
   (or ACF features) over a *region*, not all 25 layers at full locus. Even keeping the
   concat, materializing with `layers=[the_one_needed]` + a genomic window shrinks the
   object ~20-40x and makes most real queries tractable. The current default pulls
   everything -- that's the bulk of the waste.
2. **Partition-native set (the architectural fix).** A "set" becomes a **manifest of
   pointers** into the per-sample store, not a re-concatenated monolith.

### Proposed shape

A set materialization is a **manifest**, written under
`project_outputs/sets/<set>/<composition_hash>/manifest.json`, listing the member
`(experiment_id, Reference_strand, sample)` partitions (each already a pointer-or-cache
entry in the per-sample store) plus the resolved composition it was built from. No
`base.h5ad`. The composition hash still drives invalidation, but now it guards a few KB
of pointers, not a 200 GB file.

Consumers change from "load one pooled object" to "iterate/stream partitions":

- **`iter_set_partitions(project_dir, canonical_ref, *, set_name, stage, layers,
  start, end)`** -> yields per-partition AnnData slices, each loaded via the existing
  per-partition loader (`sample_analysis._load_partition_adata`, promoted to
  `sample_store`), already handling cache (legacy) vs pointer->`materialize` (modern),
  and applying `layers`/`start`/`end` projection per partition so nothing bigger than
  one partition-slice is ever resident.
- **`embedding_store`** switches `_make_features` to consume the partition iterator:
  build the feature matrix (one layer, coverage-filtered, windowed) partition by
  partition and `np.vstack` the *feature rows* (small), never the full multi-layer
  AnnData. PCA/UMAP already only need the feature matrix; this drops peak memory to
  roughly `n_reads x n_features x 4 bytes` (one layer, one window) instead of
  `n_reads x n_positions x n_layers x 8 bytes`.
- **`sample_analysis`** is already partition-native (`_load_partition_adata`); it just
  starts from the set manifest's partition list instead of `list_per_sample_partitions`.

`materialize_set` (concat-to-one-object) is **demoted to an explicit opt-in**,
`project_adata(..., layers=[...], start=..., end=...)`, for when a caller genuinely
wants one pooled AnnData *and* it fits. It grows a guardrail: estimate
`n_reads x n_positions x n_layers x dtype_size` up front and refuse (or require an
explicit `allow_large=True`) above, say, a few GB, with a message pointing at the
window/layer/iterator alternatives -- so the >200 GB write is impossible to trigger by
accident.

### What this changes in code

- `set_store.py`: `materialize_set` -> `build_set_manifest` + `iter_set_partitions`;
  drop `base.h5ad` caching. Keep `_build_composition`/`_composition_hash`/`set_label`.
- `sample_store.py`: promote `_load_partition_adata` here as a public
  `load_partition(project_dir, exp, ref, sample, *, layers, start, end)`.
- `catalog.py`: `project_adata` stays as the explicit "one pooled object" path, gains
  the size guardrail; no longer the thing `project materialize` calls by default.
- `embedding_store.py`: `_make_features` consumes `iter_set_partitions`.
- `cli`: `project materialize -o file.h5ad` keeps working for the windowed/opt-in case
  (with the guardrail); the manifest itself is what `project sample-store-list`-style
  commands and the analysis stores consume.

### Open sub-questions for this redesign

- Some external tools (scanpy, sklearn without `partial_fit`) still want one in-memory
  matrix. The iterator + per-partition feature extraction covers PCA/UMAP/periodicity;
  anything needing a true single matrix uses the guarded `project_adata` opt-in.
- Whether to keep a *tiny* pooled obs table (all reads x identity columns only, no
  layers) as a convenience index alongside the manifest -- cheap, and useful for
  filtering/grouping without touching matrices.
- Incremental PCA (`sklearn.IncrementalPCA.partial_fit` per partition) would remove the
  "all feature rows resident at once" step too, at some accuracy cost -- worth it only
  if feature matrices themselves get large.

## Open questions

- ~~Exact cached-partition format for legacy backfill~~ -- resolved: plain
  per-partition `.h5ad` via `safe_write_h5ad`, not the zarr partitioned-store code.
  That module (`partition_store.py`) documents its own dense-partition format as
  predating the raw/load split with nothing in production writing it anymore, so
  reusing it would resurrect a deprecated format rather than share a current one.
- Garbage collection: stale `composition_hash` directories will accumulate under
  `project_outputs/sets/<set_name>/` as sets grow over time. Needs a retention policy
  (keep last N, or keep until explicitly pruned) -- not designed here.
- Read-span-definition hashing for per-sample analyses needs a canonical
  serialization (so equivalent definitions expressed slightly differently still hash
  the same) -- not designed here.
- Reference-name conflicts across FASTAs (seen already: `6BALB_cJ`/`ctcf_mNanog` differ
  between the ZC-array run and the rest) still need `reference_registry.yaml` canonical
  aliasing for cross-experiment set membership to resolve cleanly; this design doesn't
  change that, but set composition hashing should incorporate the alias resolution too
  once it exists, or a canonical-name change won't invalidate stale set caches.
- UMAP `.transform()` staleness has no automatic detection here (no drift metric) --
  purely opt-in manual recompute for now.

## Suggested implementation phasing

1. **Done.** Per-sample store: modern-experiment pointers, wired into `project add`.
   `src/smftools/project/sample_store.py` (`backfill_per_sample_store`,
   `list_per_sample_partitions`), called from `cli/project_cmd.py::project_add`.
   Catalogs `(Reference_strand, Sample)` partitions with read counts as
   `pointer.json` under
   `project_outputs/per_sample/<experiment_id>/<Reference_strand>/<sample>/`. Tests:
   `tests/unit/test_project_sample_store.py`,
   `tests/unit/test_project_cli.py::test_project_add_cli_backfills_per_sample_store_for_modern_experiment`.
2. **Done.** Legacy backfill on `project add`. Same `backfill_per_sample_store`, now
   branches on `uns["is_spine"]`: legacy spines write each partition's molecules to
   an uncompressed `cache.h5ad` alongside the pointer (`kind: "cache"` vs.
   `kind: "pointer"`), read back via the new `load_per_sample_partition()`. The
   cache format chosen was a plain per-partition `.h5ad` via `safe_write_h5ad` --
   *not* the existing zarr partitioned-store code in `partition_store.py`, since
   that module's own docstring says its dense-partition format predates the
   raw/load split and nothing in production writes it anymore; reusing it here
   would have resurrected a deprecated format rather than reused a current one.
   Tests: `test_backfill_per_sample_store_caches_legacy_spine`,
   `test_backfill_per_sample_store_legacy_cache_matches_source_data`,
   `test_load_per_sample_partition_rejects_pointer_kind`,
   `test_load_per_sample_partition_missing_raises` (all in
   `tests/unit/test_project_sample_store.py`), plus
   `tests/unit/test_project_cli.py::test_project_add_cli_caches_per_sample_store_for_legacy_file`
   (also asserts the source file is never mutated).
3. **Done (base caching only -- no per-sample joins yet).** Set store:
   `src/smftools/project/set_store.py` (`materialize_set`, `set_cache_dir`). Caches
   under `project_outputs/sets/<set_name_or_canonical_reference>/<composition_hash>/
   base.h5ad` + a `composition.json` for inspection. `composition_hash` covers every
   parameter that changes the result (`canonical_reference`, `set_name`, `modality`,
   `experiments`, `stage`, `start`, `end`, `layers`, `read_metrics`) *and* the
   currently-resolved membership (`experiment`/`stage`/`spine_path`/
   `reference_strands` per matched experiment) -- registering a new experiment, or
   re-registering one with different data, changes the resolved membership and so the
   hash, which transparently falls through to a fresh materialize rather than serving
   something stale. `force_recompute=True` bypasses a cache hit outright.
   Required a small refactor first: extracted `resolve_set_members()` out of
   `project_adata` in `catalog.py` so both it and `set_store` share one
   membership-resolution implementation instead of the composition hash risking
   silent drift from what `project_adata` actually materializes.
   **Per-sample catalog joins are not implemented yet** -- there's no project-computed
   per-sample analysis catalog to join in yet either (Phase 2 only cached/pointed at
   raw molecules, not analysis outputs), so this is deferred rather than stubbed.
   No garbage collection of stale composition-hash directories (open question, as
   before). Tests: `tests/unit/test_project_set_store.py` (cache write/hit/
   force-recompute/invalidation-on-new-registration/no-match-raises/cheap-cache-dir-
   lookup), plus `tests/unit/test_project_catalog.py` and
   `tests/unit/test_project_cli.py` re-run unmodified to confirm the `catalog.py`
   refactor didn't change `project_adata`'s behavior.
4. **Done -- per-sample analysis catalog + join.**
   `src/smftools/project/sample_analysis.py`: `compute_periodicity()` wraps
   `smftools.analysis.compute.autocorrelation.compute_single_molecule_periodicity_direct`
   (Tier 2, reused as-is -- see `src/smftools/analysis/CLAUDE.md`) rather than
   reimplementing, caches its result under each per-sample-store partition's
   `analyses/periodicity/<definition_hash>/result.parquet`, keyed by every parameter
   that changes the result (layer, window, method, LS kwargs), indexed by `read_id`.
   Loads a partition's molecules from the cache (legacy) or through the registry +
   `materialize()` (modern pointer) via `_load_partition_adata`. `join_periodicity()`
   is the read side -- attaches an *already-computed* result onto a materialized
   selection by read_id (`periodicity_<column>` obs columns, NaN where uncomputed or
   filtered out); it's a separate, explicit call after `set_store.materialize_set()`
   rather than automatic inside it, since `materialize_set` has no way to know which
   analysis/definition is relevant to a given caller. `ls_freqs`/`ls_power` (the
   array-valued columns the direct method also returns) are dropped before caching --
   not parquet-safe as raw object-dtype arrays, matching that function's own
   docstring guidance. Renamed `sample_store._partition_dir` to public
   `partition_dir_for` since it's now shared across modules (no other cross-module
   import in this package uses an underscore-prefixed name, so this matches existing
   convention). Tests: `tests/unit/test_project_sample_analysis.py` (9 tests --
   compute/cache-hit/force-recompute/definition-isolation/legacy-cache-path/
   missing-partition-error/join-by-read-id/join-no-op-without-required-columns/
   join-no-op-without-cached-data), reusing the exact synthetic periodic-signal
   fixture already proven in `tests/unit/analysis/test_autocorrelation.py`.

5. **Done.** Embedding persistence: `src/smftools/project/embedding_store.py`
   (`fit_or_extend_embedding()`, `embedding_dir()`, `EmbeddingCompositionError`).
   Required extending Tier 2 `dimensionality_reduction.py` first --
   `umap_from_pca()`/`run_pipeline()` previously returned only transformed output
   arrays (`X_umap`, cluster labels), not the fitted PCA/UMAP model objects needed
   to `.transform()` new data later; confirmed no existing code called either
   function before changing their return shape, so this was safe.
   `fit_or_extend_embedding()`:
   - No existing embedding (or `force_recompute=True`): full fit via `run_pipeline`
     (PCA → UMAP → KNN → Leiden). `force_recompute` first archives whatever was
     there under `versions/<timestamp>/` rather than overwriting it.
   - Existing embedding, current set is a strict superset of what's already embedded
     (pure growth): only the new reads' features are `.transform()`ed through the
     persisted PCA/UMAP models -- existing coordinates provably don't move (asserted
     directly in tests) -- and each new point gets a cluster label via
     nearest-neighbor lookup against the fixed embedding (`sklearn.neighbors.
     NearestNeighbors`, k=1), not a Leiden recompute (no clean incremental
     equivalent -- a bigger graph can reshuffle every community, not just the new
     points).
   - Existing embedding, but some previously-embedded read is no longer present
     (shrink, or membership changed in some other way): raises
     `EmbeddingCompositionError` unless `force_recompute=True`, rather than
     silently reinterpreting the embedding.
   Also renamed `set_store._slug`/`_sets_root` to public `slug`/`sets_root`, and
   added `set_label()`, since `embedding_store` needed to agree with `set_store` on
   which directory a set's artifacts live under (same cross-module-import-uses-
   public-names convention established in the Phase 4a rename).
   Tests: `tests/unit/analysis/test_dimensionality_reduction.py` (6 tests on the
   Tier 2 change itself -- fitted models are real transformers, PCA replay is
   deterministic, UMAP transforms new points, `cluster_from_pca` unchanged),
   `tests/unit/test_project_embedding_store.py` (6 tests -- full fit, cache-hit,
   extend-without-moving-existing-points, raises-on-shrink, force-recompute-
   archives-previous, cheap-dir-lookup), using two-Gaussian-blob synthetic data
   registered through real `write_raw_store`/`add_experiment` (proving the full
   registry → set_store → embedding_store chain, not just the compute step).

   This closes Phase 4 as originally scoped in this doc. Remaining unimplemented:
   automatic per-sample-catalog joining inside `materialize_set` itself (currently
   a separate, explicit `join_periodicity()` call -- see Phase 4a note above) and
   the open questions below (GC, definition-hash canonicalization, reference-alias
   interaction with composition hashing, UMAP drift detection).

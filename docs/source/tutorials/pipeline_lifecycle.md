# Pipeline lifecycle, latent spaces, and migration

This page describes how experiment stages, project queries, latent coordinate
systems, and persisted analysis models fit together. It also records the
migration boundary for artifacts created before the current partitioned
pipeline schemas.

## Experiment and project boundaries

`smftools experiment latent` is an independently invokable experiment stage.
`smftools experiment full` also runs it after HMM by default; set
`full_run_latent: false` to stop the full workflow after HMM. A partitioned
latent run fits one independent coordinate system for each experiment,
reference, and analysis core. PCA,
UMAP, NMF, or CP axes from different owners are not aligned merely because
their representation names match.

Generic project materialization and export remain genomic operations. An
explicit `stage="latent"` or `--stage latent` request is rejected because
pooling task-local coordinates would silently discard their ownership:

- Use `smftools project export-latent` or
  `ProjectCatalog.iter_latent_parts()` to preserve one artifact per local
  experiment/core coordinate owner.
- Use the project embedding API when one coordinate system fitted across a
  multi-experiment selection is required.

Default genomic project materialization prefers the consolidated
`experiment_spine.h5ad` when it exists. That spine exposes the latest validated
pointers from sibling pipeline branches. Without it, resolution falls back to
the most-derived registered genomic stage available for each experiment.

`smftools project add` is idempotent and also acts as a refresh. Re-run it after
producing a new stage or index so the registry discovers and merges the new
spine and catalog pointers without replacing the experiment identity.

## Choosing a latent input

`latent_execution_mode` and `from_adata_stage` jointly select the input:

| Execution mode | Valid `from_adata_stage` values | Behavior |
| --- | --- | --- |
| `auto` | unset, `preprocess`/aliases, `spatial`, or `hmm` | Uses the requested partitioned spine, or prefers HMM, then spatial, then preprocess |
| `partitioned` | unset, `preprocess`/aliases, `spatial`, or `hmm` | Same partitioned selection rules as `auto`; never falls back to a monolithic file |
| `legacy` | any stage recognized by the legacy stage map | Selects a monolithic AnnData artifact and never selects a partitioned spine |

An explicit missing source fails rather than falling back to another stage.
Stages outside preprocess, spatial, and HMM are rejected in partitioned modes.
Use `legacy` only for compatibility with monolithic inputs; new latent outputs
should use the partitioned lifecycle.

## Restart, growth, and force-redo

Raw ingestion classifies the requested canonical input manifest against the
selected immutable raw generation before doing work:

- An identical manifest is a restart/cache hit and performs no extraction.
- A pure addition with unchanged raw configuration and alignment-reference
  identity processes only the added sources. A complete FASTQ pair must be
  added together.
- Removal, changed bytes at an existing source, changed sample/barcode/pair
  metadata, completing a previously unpaired source by changing its pair
  declaration, or a reference/configuration change performs a full recompute.

An append writes a new complete generation. Prior raw shards and unchanged
index pieces are checksum-matched and hardlinked directly from the selected
immutable generation; canonical working files are never used as reuse
authority. Aggregate molecule/segment catalogs and indexes are rebuilt, and
duplicate molecule or segment identities fail before the current-generation
pointer advances. The generation manifest records the transition, reused and
added source IDs, prior generation ID, and reused/new file and byte counts.
An interrupted append leaves the prior generation selected.

Partitioned latent output is published as immutable generations. A generation
becomes current only after its task stores, model bundles, indexes, plots,
spine pointers, checksums, schemas, source provenance, and completion manifest
validate.

- An exact source and compute configuration match is a restart/cache hit.
- A plot-only change reuses the validated compute generation and publishes new
  plot artifacts.
- Compatible experiment growth reuses the prior immutable model, preserves
  existing coordinates, and transforms only new molecules.
- Changed existing inputs, a source change outside verified append-only growth,
  fit membership, selected feature masks, or output-affecting model parameters
  create a new model and coordinate generation.
- `force_redo_latent_analyses=true` intentionally refits and gives the model a
  new forced-fit revision. Previous complete generations remain available.
- An interrupted or invalid replacement never changes the current pointer.

The configured fit and transform read counts are ceilings. Before material
allocations, the resource planner may lower effective fit or transform counts
to fit live memory headroom, but never below the minimum viable fit size. CP
requires a complete unit and follows `latent_cp_memory_policy=skip` or `fail`.
Every effective decision is recorded in `resource_plan.json` and the task
catalog.

## Model and cache trust boundaries

Latent task results, read-index rows, plot records, and task catalogs carry the
same model ID and checksum. Model IDs include source, coordinate owner,
representation and feature-mask identities, output-affecting parameters,
deterministic fit membership, implementation/schema versions, and an optional
forced-fit revision.

CP factors are stored as portable NumPy arrays. PCA, UMAP, and NMF estimator
state requires Python serialization, so model manifests checksum it, record
dependency versions, and mark it as trusted local data. Loading estimator
pickles requires an explicit trusted-local decision and exact dependency
compatibility. Do not exchange or load model pickle files from an untrusted
experiment or project tree.

Project embeddings use the same safety principle but are a separate product.
They publish immutable project-scoped generations. Exact coordinate reads do
not unpickle estimators; compatible extension requires
`trust_local_models=True`, while source changes, removals, or changed existing
features require `force_recompute=True`.

## Migration guide

| Existing artifact or script | Current behavior | Migration |
| --- | --- | --- |
| Periodicity cache without experiment and molecule identity | Migrated only when its registered experiment owner is unambiguous | Re-run with `force_recompute=True` if ownership is ambiguous or the cache is rejected |
| Legacy in-place project embedding directory such as one containing only `meta.json` | Never accepted as a current immutable generation | Refit with `force_recompute=True`; the old directory is retained |
| Latent output without a completed generation manifest, read index, or model provenance | Never treated as a valid restart or scoped project source | Re-run `smftools experiment latent` in `auto` or `partitioned` mode, then re-run `project add` |
| Legacy monolithic latent AnnData | Can be selected only in `legacy` experiment mode and cannot provide scoped project owners | Re-run partitioned latent analysis before using `export-latent` or `iter_latent_parts()` |
| Script calling generic project materialization/export with `stage="latent"` | Rejected with coordinate-ownership guidance | Use scoped latent export for local spaces or the project embedding API for a shared space |

Migration never reinterprets old local axes as project-global coordinates and
never infers project molecule identity from a bare read ID shared by multiple
experiments.

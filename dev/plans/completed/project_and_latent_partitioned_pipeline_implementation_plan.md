# Project and latent partitioned-pipeline implementation plan

**Plan date:** 2026-07-22

**Repository:** `smftools`

**Repository state at planning time:** `baccc3e` (`origin/main`, detached worktree)

**Source audit:** [project_and_latent_partitioned_pipeline_audit.md](project_and_latent_partitioned_pipeline_audit.md)

## Objective

Close the project/latent compatibility gaps identified after completion of the
experiment/project partitioned-pipeline program. The result must:

- preserve collision-free project molecule identity through analysis caches;
- make partitioned latent execution transactional, restart-safe, and
  resource-aware;
- make every latent molecule traceable through Parquet indexes;
- expose experiment-local latent outputs without implying that independently
  fitted coordinate systems can be pooled;
- keep project-global embeddings separate, correctly keyed, and transactionally
  published; and
- document and test the boundary between local latent artifacts and project-wide
  embeddings.

This program extends the completed PR-00 through PR-14 work. Feature branches
must be cut from current `main`, must describe the work rather than a release
version, and must not modify `src/smftools/_version.py`.

## Audit finding IDs

| ID | Severity | Finding |
|---|---|---|
| PL-C1 | Critical | Project selection of the latent stage silently omits latent representations |
| PL-C2 | Critical | Project periodicity caches and joins use bare read identity |
| PL-H1 | High | Partitioned latent lacks transactional lifecycle and restart validation |
| PL-H2 | High | Latent fit/chunk limits are not derived from estimated peak memory |
| PL-H3 | High | Project embedding cache definitions omit selection and source provenance |
| PL-H4 | High | Latent outputs have no molecule-level derived read index |
| PL-M1 | Medium | `from_adata_stage` conflicts with forced partitioned latent mode |
| PL-M2 | Medium | Latent configuration ranges are parsed but not validated |
| PL-M3 | Medium | Latent fit/model provenance and stable extension are incomplete |
| PL-M4 | Medium | Project registry refresh and documented spine selection can become stale |
| PL-M5 | Medium | Project export provenance does not identify a complete source generation |

## Agreed design contracts

These contracts constrain implementation. A PR should not change them without a
separate design review and an update to this plan.

### Experiment-local latent spaces are not project-global embeddings

Every partitioned latent task owns an independent coordinate system identified
by:

- `experiment_uid`;
- `reference`;
- `analysis_core_id`;
- core bounds;
- latent model ID; and
- representation name.

Coordinates from different owners must not be concatenated, averaged, compared
as if axes were aligned, or returned as one pooled `obsm` matrix. Shared column
names such as `X_pca` or `X_umap` do not establish a shared coordinate system.

The project embedding store is a separate product. It fits one coordinate system
over an explicitly resolved project selection and is the supported path for
cross-experiment visualization and clustering.

### Generic project materialization remains genomic

`project_adata`, `project materialize`, and partitioned project export continue
to assemble a shared genomic position axis. They must reject explicit
`stage="latent"` selection with an actionable error rather than returning
upstream-only data that appears to contain latent results.

A separate scoped latent API and export command will expose local task artifacts
without pooling coordinate axes. The API returns or writes one result per latent
coordinate owner and includes the owner metadata with every part.

### Molecule identity is never inferred from pooled observation names

Persistent and cross-experiment joins use `molecule_uid`. Where compatibility
requires the expanded identity, use `(experiment_uid, read_id)`. Bare
`read_id` remains a data field and may be used only inside a single, explicitly
owned experiment after uniqueness is verified.

One molecule may legitimately occur in several latent analysis cores. The latent
index uniqueness key is therefore:

```text
(experiment_uid, molecule_uid, analysis_core_id, representation/task generation)
```

### Completion is manifest-driven

The existence of `spine.h5ad` is not a completion signal. Partitioned latent
skip/restart decisions require a compatible completed stage manifest whose
required artifacts exist and pass validation.

All task stores, indexes, catalogs, plots, and the latent spine are built under a
unique temporary generation and atomically published. A failure leaves the prior
complete generation visible or leaves no complete generation.

### Resource limits cover model expansion, not only input rows

Latent planning estimates the input matrix plus enabled PCA, UMAP, NMF, CP,
transformation, serialization, and plotting allocations. Fit and transform
counts are derived from the resolved resource envelope and current headroom.
`latent_max_fit_reads` and `latent_transform_chunk_reads` remain user ceilings,
not guarantees that the requested size is safe.

Persisted task layout and logical coordinate ownership remain independent of the
producer's CPU and memory. Machine-specific fit/chunk decisions are recorded as
execution provenance but do not redefine task identity unless they change model
fit membership.

### Project embeddings preserve extendability without accepting stale sources

The stable embedding definition includes all semantic selection and model
parameters, but excludes the resolved membership list so a set can grow into the
same embedding.

The artifact manifest separately records each member's immutable source
fingerprint and molecule membership digest. Cache behavior is:

- identical members and source fingerprints: cache hit;
- pure addition with all existing member fingerprints unchanged: transform new
  molecules into the existing space;
- removal of a previous molecule: require explicit full refit;
- changed source fingerprint or features for an existing molecule: require
  explicit full refit; and
- incompatible definition/schema/software: create or require a new definition.

## Delivery strategy

Use one focused branch/PR per item. The main dependency chain is:

```text
PL-15 identity repair

PL-16 latent input/config contract
    -> PL-17 latent lifecycle
        -> PL-18 latent resource planning
        -> PL-19 latent molecule index
            -> PL-20 scoped project latent access

PL-21 embedding cache hardening

PL-18 + PL-19
    -> PL-22 latent model provenance

all implementation PRs
    -> PL-23 documentation and acceptance
```

PL-15, PL-16, and PL-21 can proceed independently. PL-18 and PL-19 both depend
on the final lifecycle layout from PL-17. PL-20 depends on the index contract.
PL-22 depends on the resource planner because fit membership is part of model
identity.

## Ordered PR backlog

| ID | Suggested branch | Primary outcome | Audit coverage | Depends on |
|---|---|---|---|---|
| PL-15 | `fix/project-analysis-molecule-identity` | Periodicity caches and joins use project-global molecule identity | PL-C2 | None |
| PL-16 | `fix/latent-input-config-contract` | Valid partitioned source selection and fail-fast latent config validation | PL-M1, PL-M2 | None |
| PL-17 | `feature/latent-stage-lifecycle` | Transactional latent publication and manifest-driven restart | PL-H1 | PL-16 |
| PL-18 | `feature/latent-resource-planning` | Memory-estimated fit, transform, CP, write, and plot decisions | PL-H2 | PL-17 |
| PL-19 | `feature/latent-molecule-index` | Indexed molecule-to-latent-task traceability | PL-H4, part of PL-M4 | PL-17 |
| PL-20 | `feature/project-scoped-latent-access` | Explicit local latent query/export and rejection of misleading generic materialization | PL-C1, PL-M4 | PL-19 |
| PL-21 | `feature/project-embedding-provenance` | Collision-free definitions, source validation, and atomic embedding generations | PL-H3, PL-M5 | None |
| PL-22 | `feature/latent-model-provenance` | Stable fit selection, immutable model IDs, and reproducible transform provenance | PL-M3 | PL-18, PL-19 |
| PL-23 | `feature/project-latent-acceptance` | Integrated acceptance coverage, docs, and migration guidance | All | PL-15 through PL-22 |

## PL-15 — project analysis molecule identity

### Scope

- Version the periodicity result/cache schema.
- Persist `experiment_uid`, `molecule_uid`, and original `read_id` in every result.
- Resolve the owning experiment UID before computing a per-sample analysis.
- Prefer identity already present in loaded `obs`; otherwise derive
  `molecule_uid` using the registry's experiment UID and explicit read IDs.
- Join results to project AnnData by `molecule_uid`.
- Remove cross-experiment `drop_duplicates(subset="read_id")`.
- Treat legacy cache files without project-global identity as incompatible by
  default. If a cache is unambiguously owned by one experiment, provide a
  narrowly scoped migration reader that stamps that experiment UID.

### Primary files

- `src/smftools/project/sample_analysis.py`
- `src/smftools/project/registry.py` or a small shared identity lookup helper
- `src/smftools/informatics/molecule_identity.py`
- `tests/unit/test_project_sample_analysis.py`
- project materialization fixtures used by catalog tests

### Required tests

- Periodicity attaches correctly to a real `project_adata` result whose obs names
  use pooled encoding.
- Two experiments contain the same bare `read_id`; both periodicity records join
  to the correct molecule.
- Reordering project AnnData does not change attachment.
- A missing result produces NaN only for the unmatched molecule.
- Legacy cache behavior is deterministic: migrated when ownership is
  unambiguous, otherwise rejected with remediation guidance.
- Definition hashes remain distinct for different analysis parameters.

### Exit gate

No project-level analysis join or deduplication in the touched path uses bare
`read_id` as a globally unique key.

## PL-16 — latent input and configuration contract

### Scope

- Move `latent_execution_mode` validation into experiment-config validation.
- Validate enabled latent parameters:
  - `latent_n_pcs >= 1`;
  - `latent_nmf_components >= 1`;
  - `latent_nmf_max_iter >= 1`;
  - `latent_knn_neighbors >= 2`;
  - `latent_leiden_resolution > 0`;
  - `latent_max_fit_reads >= latent_min_reads >= 2`;
  - `latent_transform_chunk_reads >= 1`;
  - `latent_cp_rank >= 1`; and
  - `latent_cp_iterations >= 1`.
- Validate only parameters relevant to enabled algorithms where disabling an
  algorithm makes its settings irrelevant.
- Add one source-resolution helper that maps `from_adata_stage` to the
  corresponding partitioned or legacy path.
- In `partitioned` mode, accept a requested preprocess/spatial/HMM stage when its
  spine exists; reject unsupported stages with the allowed names.
- In `auto` mode, honor an explicit compatible requested stage before applying
  HMM -> spatial -> preprocess preference.
- Keep legacy behavior available only through `latent_execution_mode=legacy`.
- Return precise errors for a requested stage whose artifact is absent.

### Primary files

- `src/smftools/config/experiment_config.py`
- `src/smftools/cli/latent_adata.py`
- `src/smftools/cli/helpers.py` if stage resolution is shared there
- `src/smftools/config/defaults/default.yaml`
- modality defaults only if they currently override these fields
- `tests/unit/config/test_LoadExperimentConfig.py`
- `tests/unit/test_latent_partitioned_cli.py`

### Required tests

- Every invalid boundary fails during configuration loading.
- Disabled algorithms do not require otherwise unused optional parameters.
- Partitioned mode plus each supported `from_adata_stage` selects the named
  spine.
- Auto mode with and without an explicit stage is deterministic.
- Partitioned mode never falls back to a monolithic artifact.
- Legacy mode never silently selects a partitioned spine.

### Exit gate

All invalid latent configuration and source combinations fail before executor
startup, and every supported combination resolves exactly one source artifact.

## PL-17 — transactional latent stage lifecycle

### Scope

- Wrap partitioned latent execution in the shared stage lifecycle.
- Define latent output-affecting config fields for compatibility hashing.
- Define plot-only fields separately so plot regeneration does not force model
  refitting.
- Record source spine identity, source manifest/generation, analysis planner
  version, region-catalog identities, and enabled representation configuration.
- Build a complete latent generation beneath a unique temporary directory.
- Validate before publication:
  - task catalog schema and row count;
  - every task group exists and is readable;
  - catalog `obsm_keys`, `varm_keys`, and `obs_columns` agree with stores;
  - plot catalog entries resolve;
  - latent read index when introduced by PL-19;
  - thin spine pointers resolve inside the published run; and
  - required checksums/content IDs agree.
- Publish atomically using the established stage artifact-manifest patterns.
- Regenerate the consolidated experiment spine only after latent publication is
  complete.
- Replace the CLI's `spine.exists()` skip condition with lifecycle compatibility
  validation.
- On force-redo, create a new generation; do not write through an existing live
  generation.

### Primary files

- `src/smftools/cli/latent_adata.py`
- `src/smftools/tools/partitioned_latent.py`
- `src/smftools/cli/helpers.py`
- `src/smftools/informatics/experiment_manifest.py`
- `src/smftools/informatics/sidecar_manifest.py`
- `src/smftools/informatics/experiment_spine.py`
- `tests/unit/test_latent_partitioned_cli.py`
- `tests/unit/informatics/test_experiment_manifest.py`

### Required failure-injection tests

- Failure during the first and a later task.
- Failure after task catalog creation.
- Failure during plot generation.
- Failure after thin-spine creation but before final publication.
- Missing, unreadable, or checksum-mismatched task store after a nominally
  complete run.
- Changed source/config invalidates the generation.
- Plot-only change reuses compute artifacts and republishes plot artifacts.
- Force-redo leaves no stale task groups in the new generation.
- A prior complete generation remains readable if replacement fails.

### Exit gate

No code path treats latent spine existence alone as completion. A partially
written latent generation is never selected by CLI restart or experiment-spine
publication.

## PL-18 — latent resource planning

### Scope

- Add a versioned latent estimator and resource-plan record.
- Estimate each unit using:
  - selected reads and positions;
  - source and fit matrix dtypes;
  - PCA inputs, components, transformed output, and solver workspace;
  - UMAP graph/search and transform workspace;
  - NMF input, factors, and iteration workspace;
  - CP tensor, factors, and backend expansion;
  - in-memory result `obs`/`obsm`/`varm`;
  - Zarr serialization/chunk staging; and
  - plotting reads, coordinates, colors, and loadings.
- Resolve current headroom before each fit, transform sequence, write, and plot.
- Derive effective fit reads and transform chunk reads as the smaller of the user
  ceiling and memory-safe count.
- Preserve `latent_min_reads`; fail before materialization when even the minimum
  viable fit cannot fit.
- Make CP independently skippable or fail-fast according to a documented policy.
  Default policy: if CP is enabled and cannot fit at the minimum viable unit,
  skip CP with a structured reason while allowing other enabled
  representations; an explicit strict option may turn this into failure.
- Do not silently reduce component/rank semantics to fit memory.
- Persist per-unit predicted peak, selected fit/chunk counts, limiting operation,
  skip reasons, resource envelope ID, estimator version, and measured peak.
- Keep execution sequential initially. Do not add latent worker concurrency in
  this PR.
- Bound plotting by deterministic subsampling where configured/necessary and
  preserve full-data model outputs.

### Primary files

- `src/smftools/tools/partitioned_latent.py`
- `src/smftools/memory_guard.py`
- a focused latent estimator module if needed to avoid executor growth
- `src/smftools/config/experiment_config.py`
- default configuration
- latent resource tests and integration resource-runtime tests

### Required tests

- Estimate increases monotonically with reads, positions, components, and
  enabled algorithms.
- A small envelope reduces fit and transform counts before allocation.
- A minimum unit that cannot fit fails with estimator and operation details.
- CP skip/fail policy is deterministic and recorded.
- Plot preflight limits plot materialization without changing full model output.
- Catalog and manifest record requested ceilings, effective decisions, predicted
  peak, and measured peak.
- Resource decisions do not change analysis-core identity or portable store
  layout.

### Exit gate

Every material latent allocation has a pre-allocation resource decision, and a
unit cannot exceed configured memory merely because its position axis is wide.

## PL-19 — latent molecule index

### Scope

- Extend the derived-read-index schema for latent task outputs.
- Write one index row per molecule per latent task/core with:
  - schema version;
  - `experiment_uid`;
  - `molecule_uid`;
  - original `read_id`;
  - reference and optional canonical reference UID;
  - `analysis_core_id`, core start, and core end;
  - latent task/model generation ID;
  - relative group path;
  - group row;
  - available representation/label summaries; and
  - task/model checksum or content ID.
- Partition the Parquet dataset for useful pruning without creating one file per
  molecule.
- Validate uniqueness by molecule/core/generation, not molecule alone.
- Add the latent index pointer to the thin and consolidated experiment spines.
- Teach project registry discovery to register `latent_read_index`.
- Teach `ProjectCatalog.lookup_molecule` to return all matching latent core rows.
- Preserve index/path portability after moving an experiment and project tree.

### Primary files

- `src/smftools/informatics/derived_read_index.py`
- `src/smftools/tools/partitioned_latent.py`
- `src/smftools/informatics/experiment_spine.py`
- `src/smftools/project/registry.py`
- `src/smftools/project/catalog.py`
- latent, derived-index, project catalog, and relocation tests

### Required tests

- One molecule in multiple analysis cores returns multiple correctly scoped rows.
- Duplicate bare read IDs across experiments remain distinct.
- Lookup uses Parquet indexes without opening task Zarr stores.
- Predicate pruning selects only required index partitions.
- Missing/duplicate/out-of-range `group_row` fails publication validation.
- Registry re-add discovers the index.
- Consolidated-spine and relocated-project lookup both resolve the index.

### Exit gate

One project molecule can be traced from raw through latent using indexes only,
with every returned latent record carrying unambiguous coordinate ownership.

## PL-20 — scoped project latent access

### Public behavior

Add a scoped reader, tentatively:

```python
ProjectCatalog.iter_latent_parts(
    *,
    canonical_reference=None,
    experiments=None,
    set_name=None,
    molecule_uids=None,
    analysis_core_ids=None,
    representations=None,
    labels=None,
)
```

Each yielded item contains one task-local AnnData/result plus immutable scope
metadata. The reader may project rows and requested `obsm`/`varm`/`obs` fields,
but it must not concatenate different task owners.

Add a project CLI export, tentatively:

```text
smftools project export-latent PROJECT_DIR OUTPUT_DIR [filters...]
```

It writes a portable partitioned directory with one local coordinate artifact
per owner and a top-level catalog. The export catalog records experiment, core,
model, representation, source checksum, and row count.

### Generic materialization behavior

- Reject `stage="latent"` in `project_adata`, generic project materialization,
  and genomic project export.
- The error directs callers to scoped latent access for experiment-local outputs
  or the project embedding command/API for a shared project coordinate system.
- Continue allowing default consolidated experiment spines; their latent
  pointers do not cause latent data to be attached implicitly.
- Do not add latent fields to `partition_read.materialize`.

### Primary files

- `src/smftools/project/catalog.py`
- a focused `src/smftools/project/latent_store.py`
- `src/smftools/project/registry.py`
- `src/smftools/cli/project_cmd.py`
- `src/smftools/cli_entry.py`
- project/latent CLI and catalog tests

Before editing CLI files, re-read `src/smftools/cli/AGENTS.md`.

### Required tests

- Explicit generic `stage="latent"` fails with actionable guidance.
- Default experiment-spine genomic materialization remains unchanged.
- Row and representation projection occurs before task materialization.
- Two local owners are yielded/exported as two scoped artifacts.
- No API returns a combined `X_pca`/`X_umap` across owners.
- Requested molecule IDs use the latent index and touch only matching task
  stores.
- Export is transactional, relocatable, and rejects an existing target.
- Legacy monolithic latent artifacts either use an explicitly documented
  compatibility reader or fail with clear migration guidance.

### Exit gate

Project users cannot accidentally interpret independent latent coordinates as a
shared space, and they have a bounded, indexed path to retrieve the local
artifacts intentionally.

## PL-21 — project embedding provenance and atomic generations

### Definition schema

Version the stable embedding definition and include:

- canonical reference identity;
- named-set identity or explicit selection identity;
- modality filter;
- explicit experiment filter;
- selected pipeline stage;
- feature kind, layer, genomic window;
- PCA/UMAP/Leiden parameters;
- random state;
- relevant analysis implementation/schema versions; and
- identity schema version.

Do not include current resolved membership in the stable path hash.

### Source-generation schema

Record separately:

- resolved experiment IDs and experiment UIDs;
- selected spine/stage generation IDs and checksums;
- relevant task/catalog/config fingerprints;
- ordered molecule membership digest;
- feature-input digest or a deterministic source fingerprint sufficient to
  detect changed values for existing molecules;
- dependency versions required to read/transform the fitted models; and
- fit/extension timestamps and prior generation ID.

### Transactional layout

Publish immutable generation directories and switch a small atomic current
pointer/manifest only after all arrays, models, names, clusters, and metadata
validate. Readers resolve only a complete current generation. A failed extension
leaves the prior generation current.

Pickled estimator artifacts are trusted local artifacts only. Record package
versions and checksums, never load them from an untrusted project tree without an
explicit trust boundary, and document that they are not a stable interchange
format.

### Primary files

- `src/smftools/project/embedding_store.py`
- shared artifact/atomic helpers
- project set/member resolution helpers
- `tests/unit/test_project_embedding_store.py`
- project relocation/provenance tests

### Required tests

- No collision across canonical reference, stage, modality, explicit experiment
  filter, or named-set identity.
- Exact selection/source is a cache hit.
- Pure growth transforms only new molecules and preserves old coordinates.
- Removal requires force-refit.
- Changed source data for an existing molecule requires force-refit even when
  names are unchanged.
- Dependency/schema incompatibility produces a precise error or new definition.
- Interrupted initial fit publishes nothing.
- Interrupted extension leaves the prior generation current.
- Forced refit archives/preserves the prior complete generation.
- Relocation preserves valid relative references.

### Exit gate

No embedding cache hit or extension can cross semantic selection boundaries or
accept changed features for an existing molecule.

## PL-22 — latent model provenance and deterministic transformation

### Scope

- Select fit membership by deterministic hash ranking over
  `(experiment_uid, molecule_uid, latent_random_state, coordinate owner)` rather
  than current obs order.
- Define an immutable latent model key containing:
  - source generation and analysis-core identity;
  - representation type and selected feature mask identity;
  - output-affecting algorithm parameters;
  - fit-membership digest;
  - implementation/schema version; and
  - intentional forced-fit revision.
- Persist fitted PCA, UMAP, and NMF transform state required to reproduce or
  extend coordinates. Persist CP factors and fit metadata even when CP is not
  transformable under the same contract.
- Prefer portable arrays/JSON for model state where practical. If an estimator
  serialization is required, checksum it, record dependency versions, mark it as
  trusted local data, and validate compatibility before loading.
- Record model ID/checksum on every task, index row, task-catalog row, and plot
  source record.
- On compatible experiment growth, reuse the prior model and transform only new
  molecules. Existing coordinates must not move.
- A changed source, fit membership, feature mask, or output-affecting parameter
  creates a new coordinate generation.
- A full refit must be explicit and preserve the previous complete generation.

### Primary files

- `src/smftools/tools/partitioned_latent.py`
- a focused latent model artifact module
- latent task/index/catalog schemas
- lifecycle compatibility definitions
- latent executor and provenance tests

### Required tests

- Fit membership is invariant to task/read input ordering.
- Same source/config produces the same semantic model ID.
- Existing coordinates remain byte/numerically stable after compatible growth.
- Only new molecules are transformed during extension.
- Changed source/config/mask produces a new generation.
- Tampered or version-incompatible model artifacts are rejected.
- Every result, index row, and plot can be traced to one model ID.
- CP provenance is complete even when incremental transformation is unsupported.

### Exit gate

Every latent coordinate can be traced to immutable source, scope, fit membership,
configuration, and model artifacts, and compatible growth does not redefine
existing coordinates.

## PL-23 — documentation and integrated acceptance

### Documentation

Update user-facing documentation to state:

- latent remains an independently invokable experiment stage and runs after
  HMM by default in `smftools experiment full`, with an explicit opt-out;
- each experiment/reference/core latent space is independent;
- generic project materialization/export is genomic and rejects an explicit
  latent stage;
- scoped project latent export preserves local spaces;
- project embeddings are separately fitted shared spaces;
- default project materialization prefers the consolidated experiment spine;
- `project add` refreshes explicitly discovered stage/index pointers;
- valid `from_adata_stage` and execution-mode combinations;
- restart and force-redo behavior;
- memory ceilings may lower effective fit/chunk counts; and
- cache/model artifacts have version and trust boundaries.

Add migration notes for:

- legacy periodicity caches;
- old embedding directories;
- latent outputs without completed manifests/read indexes/model provenance; and
- scripts that previously passed `stage="latent"` to generic project
  materialization.

Before editing documentation, re-read `docs/source/AGENTS.md` and run the
warnings-as-errors documentation build.

### Acceptance matrix

At minimum cover:

| Dimension | Values |
|---|---|
| Experiment count | one, two with duplicate bare read IDs |
| Analysis mode | locus, genome |
| Latent source | preprocess, spatial, HMM |
| Execution mode | auto, partitioned, legacy compatibility/error path |
| Full recipe | latent disabled, latent enabled after HMM |
| Representations | PCA/UMAP, NMF, CP enabled/disabled combinations |
| Resource profile | roomy, fit-reducing, minimum-unit failure |
| Lifecycle | fresh, compatible restart, config change, source change, injected failure, force-redo |
| Project operation | genomic materialize, partitioned genomic export, scoped latent export, project embedding fit/extend/refit |
| Filesystem | original tree, relocated experiment/project |
| Identity | unique read IDs, duplicate read IDs across experiments, molecule in multiple cores |

### Required commands

Use the canonical `venvs/venv-all` interpreter unless the active environment
already satisfies the applicable tests:

```text
venvs/venv-all/bin/python -m pytest -q <focused project/latent/config/index tests>
venvs/venv-all/bin/python -m pytest -m unit -q
venvs/venv-all/bin/python -m pytest -m integration -q
venvs/venv-all/bin/ruff check .
venvs/venv-all/bin/ruff format --check .
sphinx-build -W -b html docs/source docs/_build/html
```

Run smoke/E2E subsets when a touched CLI path has relevant coverage. Record
optional-dependency or platform exclusions rather than weakening assertions.

### Exit gate

All audit findings have an automated acceptance criterion or an explicitly
owner-approved deferred validation. Documentation and CLI help describe actual
behavior, and the full focused project/latent suite passes after relocation.

## Schema and migration policy

- Version every changed persistent schema: periodicity results, latent task
  catalog, latent read index, latent stage manifest, project embedding
  definition, and project embedding generation manifest.
- Readers may support an older schema only when ownership and semantics can be
  recovered without guessing.
- Never reinterpret old local latent coordinates as project-global coordinates.
- Never backfill missing project identity from a bare read ID across more than
  one experiment.
- Old embedding artifacts whose semantic selection/source cannot be proven are
  stale inputs, not cache hits.
- Migration should preserve prior complete artifacts until the replacement
  generation validates.
- User-facing schema/config/CLI changes require migration notes.

## Explicit non-goals

- Aligning independently fitted latent spaces across experiments or cores.
- Treating identically named PCA, UMAP, NMF, or CP axes as equivalent.
- Adding latent worker concurrency before memory planning is measured and stable.
- Making generic genomic project materialization carry arbitrary task-local
  `obsm`/`varm`.
- Refactoring unrelated project stores or analysis algorithms.
- Replacing the project embedding algorithms.
- Introducing a new release version on feature branches.

## Program completion definition

The audit is addressed when all of the following are true:

1. Project analysis caches join by collision-free molecule identity.
2. Invalid latent configuration and source combinations fail before execution.
3. Latent completion and restart are manifest-driven and transactional.
4. Fit, transform, CP, write, and plot allocations are preflighted against the
   resource envelope.
5. A project molecule can be traced to every latent task/core through Parquet
   indexes only.
6. Generic project materialization cannot silently omit explicitly requested
   latent results.
7. Scoped latent access preserves coordinate ownership and never pools
   independent axes.
8. Project embedding cache definitions and source generations cannot collide or
   accept stale features.
9. Every latent coordinate is traceable to immutable source, scope, fit
   membership, configuration, and model provenance.
10. Focused, unit, integration, relocation, failure-injection, lint, format, and
    documentation gates pass, with any deferred validation explicitly recorded.

## Implementation status

| PR | Status | Branch | Notes |
|---|---|---|---|
| PL-15 | Implemented and committed | `fix/project-analysis-molecule-identity` | Periodicity cache schema v2 and molecule-UID joins implemented; focused validation passed in `1cc60b1` |
| PL-16 | Implemented and committed | `fix/latent-input-config-contract` | Latent config validation and deterministic partitioned/legacy source resolution implemented; validation passed in `793cf05` |
| PL-17 | Merged | `feature/latent-stage-lifecycle` | Merged to `main` in `605a8cd`; transactional immutable generations, compatibility-aware restart, and failure-injection coverage implemented in `980ad27` |
| PL-18 | Merged | `feature/latent-resource-planning` | Merged to `main` in `3e59d39`; versioned allocation estimates, live decisions, CP policy, and resource provenance implemented in `11700b4` |
| PL-19 | Implemented and committed | `feature/latent-molecule-index` | Generation-scoped latent molecule index, publication validation, spine/registry discovery, and portable project lookup implemented in `d7a7995` |
| PL-20 | Merged | `feature/project-scoped-latent-access` | Merged to `main` in `c6a1799`; scoped indexed latent reader/export and generic latent-materialization guard implemented in `20a6255` |
| PL-21 | Merged | `feature/project-embedding-provenance` | Merged to `main` in `98738c6`; collision-free definitions, source validation, trusted-model boundary, and atomic immutable generations implemented in `d4f9f58` |
| PL-22 | Merged | `feature/latent-model-provenance` | Merged to `main` in `9727a54`; deterministic fit membership, immutable model artifacts, append-only growth reuse, and end-to-end model traceability implemented in `4de5da8` |
| PL-23 | Implemented; awaiting commit | `feature/project-latent-acceptance` | Lifecycle/migration guidance, acceptance mapping, and default-on post-HMM latent orchestration with an explicit opt-out are validated |

### PL-15 implementation record — 2026-07-27

- Versioned periodicity cache results as schema 2 and persisted
  `experiment_uid`, `molecule_uid`, and original `read_id`.
- Resolved cache ownership through the project registry before computation or
  migration.
- Migrated unambiguously owned legacy caches and rejected partial, conflicting,
  duplicate, or unsupported identity records with force-recompute guidance.
- Joined pooled project results by `molecule_uid`; pooled observation names and
  bare read IDs are no longer treated as cross-experiment identity.
- Added real pooled-materialization, reordered-observation, duplicate bare-read
  ID, missing-result, explicit-identity, and legacy-cache migration coverage.
- Validation: 72 focused project tests passed; repository-wide Ruff check and
  format checks passed; warning-as-error Sphinx build passed with network
  inventories disabled.
- Full unit run: 985 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied `SC_SEM_NSEMS_MAX` access.
- Commit: `1cc60b1` (`fix: use molecule identity for project periodicity`).

### PL-16 implementation record — 2026-07-27

- Added load-time validation for `latent_execution_mode`, shared read limits,
  transform chunk size, and enabled PCA/UMAP, NMF, and CP settings.
- Disabled algorithms no longer require valid settings that execution will not
  consume.
- Centralized latent source resolution so explicit preprocess, spatial, and HMM
  requests select their named partitioned spines; aliases resolve to the
  canonical preprocess stage.
- Made automatic resolution deterministic (`hmm` -> `spatial` -> `preprocess`)
  while honoring a compatible explicit stage first.
- Restricted monolithic artifact selection and skip behavior to explicit
  `latent_execution_mode=legacy`; automatic and partitioned modes never fall
  back to legacy artifacts.
- Added precise missing-artifact and unsupported-stage errors with source-mode
  coverage.
- Validation: 63 focused config/latent tests and 92 smoke tests passed;
  repository-wide Ruff check and format checks passed; warning-as-error Sphinx
  build passed with network inventories disabled.
- Full unit run: 1,009 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied `SC_SEM_NSEMS_MAX` access.
- Commit: `793cf05` (`fix: validate latent inputs and source selection`).

### PL-17 implementation record — 2026-07-27

- Wrapped partitioned latent execution in the shared stage lifecycle and
  separated compute-affecting configuration from plot-only compatibility.
- Added source spine, upstream generation, region-catalog, planner, and enabled
  representation identities to lifecycle and generation provenance.
- Built latent results in unique staging directories, validated complete
  generations, and atomically published immutable generation directories,
  canonical spines, and current-generation pointers.
- Replaced spine-existence restart behavior with manifest, source, config,
  generation, artifact, and plot compatibility checks.
- Reused validated compute artifacts for plot-only changes while compute or
  source changes and force-redo create clean generations.
- Preserved prior complete generation metadata and readability when replacement
  fails; the experiment spine is regenerated only after lifecycle completion.
- Added failure injection for first/later task, catalog, plotting, and
  pre-publication failures plus missing, unreadable, and checksum-corrupted
  stores.
- Validation: 63 focused lifecycle/manifest tests, 63 partitioned CLI wrapper
  tests, and 92 smoke tests passed; repository-wide Ruff check and format
  checks passed; warning-as-error Sphinx build passed with network inventories
  disabled.
- Full unit run: 1,026 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied `SC_SEM_NSEMS_MAX` access.
- Commit: `980ad27` (`feat: make latent publication transactional`).

### PL-18 implementation record — 2026-07-27

- Added versioned, monotonic latent memory estimates for fit, transform, CP,
  retained results, Zarr staging, and plotting using read/position counts,
  source and fit dtypes, enabled algorithms, components, rank, and neighbors.
- Resolved live headroom before fit materialization, transform sequences,
  complete-unit CP, writes, and plots; effective fit and transform counts are
  bounded by both configured ceilings and estimator-safe counts.
- Preserved `latent_min_reads` as a hard viability boundary with estimator,
  operation, predicted-memory, and live-headroom details on failure.
- Added `latent_cp_memory_policy` (`skip` by default or `fail`) with structured
  skip reasons and no automatic component/rank reduction.
- Added deterministic lazy plot materialization bounded by
  `latent_plot_max_reads`, without changing full stored model outputs.
- Published task-catalog schema 3 plus a versioned `resource_plan.json`
  containing requested/effective decisions, limiting operations, CP reasons,
  envelope IDs, estimator version, and predicted/measured peaks.
- Kept resource ceilings outside compute identity, separated plot limits into
  plot compatibility, and preserved analysis-core IDs and portable group paths.
- Added estimator, config, executor, lifecycle, lazy-materialization, and live
  runtime integration coverage plus user-facing configuration migration notes.
- Validation: 118 focused unit/integration tests passed with 2 sandbox skips;
  92 smoke tests passed with 1 skip; repository-wide Ruff check and format
  checks passed; warning-as-error Sphinx build passed with network inventories
  disabled.
- Full unit run: 1,046 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied semaphore/forkserver access.
- Commit: `11700b4` (`feat: plan latent resources before allocation`).

### PL-19 implementation record — 2026-07-27

- Added latent read-index schema 2 with one row per molecule/core/generation,
  stable experiment and molecule identities, original read ID, reference UID,
  core ownership, generation and publication-safe group paths, stored row,
  representation/loading/label summaries, and task checksum.
- Partitioned the Parquet dataset by a stable two-hex molecule bucket with at
  most one shard per occupied task bucket; project lookup applies both molecule
  and bucket predicates for partition pruning.
- Published the immutable generation index through thin-spine, sidecar,
  generation-manifest, lifecycle, and consolidated experiment-spine pointers.
- Added publication validation for schema, molecule/core/generation uniqueness,
  generation/path/checksum ownership, identity and stored-order agreement, and
  missing, duplicate, or out-of-range group rows.
- Added registry discovery and refresh for `latent_read_index`; project molecule
  lookup returns every matching latent core without opening task Zarr stores.
- Added multi-core, duplicate bare-read ID, predicate-pruning, corruption,
  registry refresh, consolidated-spine, index-only lookup, and relocated-project
  coverage.
- Validation: 98 focused tests and 92 smoke tests passed; focused Ruff and
  format checks and warning-as-error Sphinx build passed.
- Full unit run: 1,050 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied `SC_SEM_NSEMS_MAX` access. The unsandboxed rerun
  reached 1,066 passed with 4 unrelated spawn-path failures where workers could
  not import the `tests` package.
- Commit: `d7a7995` (`feat: index latent molecules by analysis core`).

### PL-20 implementation record — 2026-07-27

- Added `ProjectCatalog.iter_latent_parts()` to retrieve one projected AnnData
  result per immutable experiment/core coordinate owner without concatenating
  independently fitted representations.
- Applied molecule, experiment/set, reference, and core filters through the
  partitioned latent read index before opening task stores; row, representation,
  loading, and label projections occur before task materialization.
- Added transactional `smftools project export-latent` output with one portable
  Zarr artifact per owner, a provenance-rich Parquet catalog, a checksummed
  completion manifest, relocation validation, and existing-target rejection.
- Rejected explicit `stage="latent"` requests at generic genomic project
  materialization and export boundaries with guidance to scoped local access or
  the shared project embedding workflow; default consolidated genomic
  materialization remains unchanged.
- Added legacy monolithic-latent migration guidance and user documentation for
  the ownership boundary and new command.
- Validation: 38 focused project/latent tests and 92 smoke tests passed;
  repository-wide Ruff check and format checks and the warning-as-error Sphinx
  build passed.
- Full unit run: 1,056 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied `SC_SEM_NSEMS_MAX` access.

### PL-21 implementation record — 2026-07-27

- Versioned the stable embedding definition across canonical reference, named
  set and explicit experiment selection, modality, selected stage,
  feature/layer/window, PCA/UMAP/Leiden settings, random state, implementation,
  and molecule-identity schema without hashing resolved membership.
- Added generation-scoped source provenance with resolved experiment identities,
  portable selected-spine paths and checksums, stage generation/config/schema
  fingerprints, ordered molecule membership, per-molecule feature digests,
  aggregate feature-input identity, and model dependency versions.
- Replaced in-place embedding updates with validated immutable generations,
  checksummed artifacts and manifests, staging cleanup, and an atomic relative
  `current.json` pointer. Initial and extension publication failures leave no
  new visible or orphaned generation.
- Exact cache hits validate and read arrays without unpickling models. Pure
  growth transforms only new molecules after the caller explicitly accepts the
  trusted-local pickle boundary; existing coordinates remain unchanged.
- Removal, reordered membership, changed feature values, and changed existing
  source artifacts require an explicit refit. Forced refits preserve all prior
  immutable generations, and dependency/schema incompatibility fails with
  actionable guidance.
- Added a relocation-safe coordinate reader and documented that estimator
  pickles are trusted local runtime artifacts rather than an interchange
  format.
- Validation: 44 focused project tests and 92 smoke tests passed;
  repository-wide Ruff check and format checks and the warning-as-error Sphinx
  build passed.
- Full unit run: 1,062 passed, 9 skipped, and 20 sandbox-only multiprocessing
  failures caused by denied `SC_SEM_NSEMS_MAX` access.

### PL-22 implementation record — 2026-07-27

- Replaced order-dependent latent fit sampling with deterministic hash ranking
  over experiment identity, molecule identity, random state, and coordinate
  owner; the ordered membership digest is part of the semantic model key.
- Added immutable model bundles keyed by source generation, analysis core,
  representation and feature-mask identities, output-affecting parameters,
  fit membership, schema/implementation versions, and intentional forced-fit
  revisions.
- Persisted checksummed PCA, UMAP, and NMF estimator state behind an explicit
  trusted-local pickle boundary with exact dependency validation. CP factors
  are stored as portable NumPy arrays with fit metadata and explicit
  non-transformable provenance.
- Added per-molecule model-input digests. Compatible append-only experiment
  growth reuses the prior immutable model, copies existing coordinates exactly,
  and transforms only new molecules; changed existing inputs, fit membership,
  masks, or algorithm parameters trigger a fresh model.
- Propagated model ID and checksum through task results, task-catalog schema 4,
  latent read-index schema 3, project scoped-owner validation, plot catalog
  records, generation schema 2, thin-spine pointers, and sidecar manifests.
- Preserved prior complete generations for explicit force-refits and retained
  plot-only generation reuse with model bundles.
- Added deterministic-order, semantic-identity, source/config/mask revision,
  tamper, trust, dependency-version, portable CP, and stable-growth coverage.
- Validation: 74 focused latent/project tests and 92 smoke tests passed;
  focused Ruff and format checks and the warning-as-error Sphinx build passed.
  The unsandboxed full unit suite passed with 1,087 tests and 9 skips.

### PL-23 implementation record — 2026-07-27

- Added a user-facing pipeline lifecycle and migration guide covering the
  independently invokable latent stage, experiment/reference/core coordinate ownership,
  partitioned versus legacy source selection, consolidated project spines,
  registry refresh, restart/growth/force-redo behavior, resource ceilings,
  immutable model provenance, and trusted-local estimator boundaries.
- Documented deterministic migration behavior for legacy periodicity caches,
  in-place project embedding directories, incomplete or monolithic latent
  artifacts, and scripts that requested latent through generic genomic project
  materialization/export.
- Added a contributor acceptance map connecting every experiment-count,
  analysis-mode, source, execution-mode, representation, resource, lifecycle,
  project-operation, filesystem, and identity dimension to focused automated
  coverage.
- Added registry acceptance proving that re-running `project add` discovers a
  newly published latent stage and generation-scoped read index without
  duplicating or replacing the experiment.
- Added relocated multi-experiment latent acceptance with duplicate bare read
  IDs, distinct project molecule identities, and one molecule participating in
  multiple independent analysis cores.
- Validation: 172 focused project/latent/config/index tests, 92 smoke tests,
  and the unsandboxed integration suite (6 passed, 2 platform/fixture skips)
  passed. The unsandboxed full unit suite passed with 1,089 tests and 9 skips.
  Repository-wide Ruff and format checks and the warning-as-error Sphinx build
  passed.
- Amended `experiment full` to run latent after HMM by default, with
  `full_run_latent: false` as the explicit opt-out. Full-workflow summary
  schema 2 records the latent stage, including a `disabled` outcome, and
  partitioned completion now validates the latent generation before
  publication.
- Promoted the UMAP, Leiden, and tensor dependencies used by the default
  latent configuration into the base runtime profile so a normal install
  continues to support the complete default `experiment full` workflow.
- Added default/false config parsing, five-stage ordering, four-stage opt-out,
  and latent failure-propagation coverage. The focused full/config/latent
  suite passed with 91 tests.
- Final validation passed with 1,092 unit tests and 9 skips, 6 integration
  tests and 2 skips, 92 smoke tests and 1 skip, repository-wide Ruff and
  format checks, and the warning-as-error Sphinx build.

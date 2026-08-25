# Project and latent compatibility audit for the partitioned pipeline

**Audit date:** 2026-07-22  
**Repository state reviewed:** `baccc3e` (`origin/main`, detached worktree)  
**Scope:** project registry/catalog/materialization/export/analysis stores and
`smftools experiment latent`, as they interface with the partitioned experiment
pipeline merged through PR-12.

## Executive assessment

The new experiment pipeline and the existing project layer are compatible for
their primary shared contract: a project can discover thin experiment spines,
harmonize references, query molecule indexes, materialize bounded genomic
projections, and export those projections transactionally. The consolidated
experiment spine also preserves the latent task-catalog pointer.

The integration is not complete for latent results themselves. A project can
register and explicitly select a `latent` spine, but project materialization does
not read the latent task catalog or attach latent `obsm`, `varm`, or latent-derived
`obs` fields. It therefore returns the upstream genomic data represented by that
spine, not the experiment-local latent representation. This is particularly
important because each latent reference/core is fitted independently and cannot
be naively concatenated into a cross-experiment coordinate system.

The safest architectural contract is:

- experiment latent outputs remain **experiment- and analysis-core-local**;
- the project embedding store remains the **project-global fitted coordinate
  space**; and
- project APIs expose those two products under different names and never imply
  that their coordinates are interchangeable.

Before claiming full compatibility, the package should also close gaps in
project identity joins, embedding-cache provenance, latent publication/restart
safety, latent resource planning, and latent molecule-level indexing.

## Current data flow

```text
raw spine/store/index
  -> preprocess partitions/read index
  -> spatial metrics
  -> HMM partitions/read index
             \
              -> latent task Zarrs (one independent space per reference/core)
                  + task_catalog.parquet
                  + thin latent spine containing pointers
                          |
all stage spines ----------> consolidated experiment spine
                                  |
                           project registry/catalog
                                  |
                    partition_read.materialize()
                     + preprocess/HMM layers
                     + optional spatial metrics
                     - latent task representations (not consumed)

project embedding store
  <- separately materializes a project selection
  <- separately fits/extends PCA, UMAP, and Leiden
  -> project-global coordinates unrelated to experiment latent coordinates
```

The experiment-spine union explicitly includes latent
(`src/smftools/informatics/experiment_spine.py:50-60`) and the project registry
discovers a `latent` stage (`src/smftools/project/registry.py:46-68`). However,
the generic materializer only assembles preprocess/HMM positional layers and
spatial read metrics (`src/smftools/informatics/partition_read.py:958-1078`).
The project iterator delegates directly to that materializer
(`src/smftools/project/set_store.py:117-167`).

## Compatibility matrix: project functionality

| Contract | State | Evidence and implication |
|---|---|---|
| Thin-spine discovery | Compatible | All principal stages plus the consolidated experiment spine are discoverable; registry paths are serialized portably (`project/registry.py:46-68`, `109-181`). |
| Reference harmonization | Compatible | Project selection resolves canonical sequence identities back to experiment reference strands before materialization. Existing project catalog tests cover cross-experiment pooling. |
| Global molecule identity | Mostly compatible | Pooled parts receive `experiment_uid`, `molecule_uid`, and collision-free obs names (`project/set_store.py:77-96`). Some downstream analysis caches still join on bare read IDs. |
| Bounded query/materialization | Compatible with explicit limits | The project estimates the whole pooled allocation before reading, enforces the resource envelope, and streams projected members before final concatenation (`project/catalog.py:554-657`). Final pooled AnnData is still intentionally in memory. |
| Partitioned project export | Compatible | Export estimates each member, chunks indexed selections, publishes through a temporary directory, and records a resource envelope (`project/catalog.py:660-735` and following). |
| Derived preprocess/HMM layers | Compatible | Derived task catalogs and read indexes are queried and stitched by reference/read/position (`informatics/partition_read.py:535-654`). |
| Spatial per-read metrics | Compatible when requested | Materialization has an explicit `read_metrics` opt-in and overlays spatial outputs. |
| Latent stage discovery | Partially compatible | Registry and experiment spine carry the stage/pointers, but latent read indexes are not discovered (`project/registry.py:187-216`). |
| Latent representations | Incompatible | Project materialization never consumes `latent_task_catalog`; `--stage latent` does not return task `obsm`, `varm`, or latent labels. |
| Project-global embedding | Functionally separate | The embedding store refits/extends PCA/UMAP/Leiden over a project materialization (`project/embedding_store.py:216-380`). It is not an aggregation of experiment latent outputs. |
| Project analysis joins | Identity defect | Periodicity results are cached and deduplicated by bare `read_id`, then reindexed against pooled obs names (`project/sample_analysis.py:111-192`, `195-253`). |
| Cache provenance | Incomplete | The project embedding definition omits selection and source identity (`project/embedding_store.py:60-108`, `216-287`). |

## Compatibility matrix: `smftools experiment latent`

| Contract | State | Evidence and implication |
|---|---|---|
| Partitioned source selection | Mostly compatible | `auto` prefers HMM, then spatial, then preprocess spines (`cli/latent_adata.py:174-192`). See the `from_adata_stage` conflict below. |
| Analysis-region ownership | Compatible | Units inherit the shared analysis-core plan and preserve analysis-region/original-coordinate metadata. Tests compare latent units with the shared planner. |
| Coordinate semantics | Explicit and correct | Each result records `independent_coordinate_system=True` and its reference/core scope (`tools/partitioned_latent.py:570-578`). |
| Bounded fit population | Partially compatible | PCA/UMAP/NMF fit at most `latent_max_fit_reads`; remaining reads transform in chunks (`tools/partitioned_latent.py:451-568`). Width and algorithm expansion are not included in a memory preflight. |
| CP behavior | Bounded by read count only | CP is skipped when the complete unit exceeds `latent_max_fit_reads` (`tools/partitioned_latent.py:493-508`), but eligible wide units still lack a memory estimate. |
| Concurrency | Compatible baseline | Units execute sequentially to avoid multiplying model memory (`tools/partitioned_latent.py:629-639`). |
| Resource-envelope enforcement | Incomplete | Post-hoc process-tree samples are recorded, but latent does not reserve headroom or derive fit/chunk limits from the resolved envelope. |
| Artifact publication | Incomplete | Task stores, catalogs, plots, spine, and sidecar are written directly into the final directory (`tools/partitioned_latent.py:579-681`), without the shared stage lifecycle or transactional publication. |
| Restart validation | Incompatible with pipeline lifecycle contract | The CLI skips whenever the latent spine exists, without validating task stores, manifest completion, checksums, or configuration compatibility (`cli/latent_adata.py:158-172`). |
| Molecule traceability | Incomplete | No latent derived-read index maps molecule identity to task/group rows. The task catalog describes units, not individual molecules. |
| Plot-region behavior | Limited by design | Plots are generated per fitted analysis core. Plot-region catalogs are not used to stitch latent results; independently fitted core coordinates should not be stitched as if shared. |
| Configuration parsing | Present but under-validated | Defaults and scalar parsing exist (`config/experiment_config.py:1239-1253`, `2335-2355`), but positivity/range constraints are not enforced at load time. |

## Findings

### Critical: selecting a project latent stage silently omits latent results

`execute_partitioned_latent` stores representations in task Zarr `obsm`/`varm`
and latent labels in task `obs`, then publishes a thin spine containing pointers
(`tools/partitioned_latent.py:519-610`, `656-665`). The consolidated experiment
spine retains those pointers, and project registration accepts the latent spine.

No project read path follows those pointers. `partition_read.materialize` supports
preprocess/HMM layers and spatial metrics only. Consequently:

- `project materialize --stage latent` returns genomic matrices and overlays
  available from upstream stages, not latent embeddings;
- `project export --stage latent` likewise exports upstream genomic data; and
- the command can appear successful while omitting the feature that motivated
  selection of the latent stage.

This should not be fixed by concatenating `X_umap` or `X_pca` from task stores.
The executor explicitly declares each reference/core coordinate system
independent. A project latent reader should expose task-local results with their
scope, while project-wide comparisons should use a separately fitted project
embedding.

### Critical: project periodicity joins violate the pooled identity contract

`compute_periodicity` derives its cached `read_id` from `adata.obs_names`
(`project/sample_analysis.py:181-192`). `join_periodicity` then concatenates all
matching caches, drops duplicates by bare `read_id`, and reindexes against the
target `adata.obs_names` (`project/sample_analysis.py:230-252`).

Project normalization changes obs names to a collision-free pooled encoding and
stores bare read identity separately, along with `experiment_uid` and
`molecule_uid` (`project/set_store.py:77-96`). The current join therefore has two
failure modes:

1. cached bare IDs do not match pooled obs names, producing missing joins; and
2. the same instrument read ID in two experiments is deduplicated incorrectly.

Persist and join on `molecule_uid`, or on the explicit composite
`(experiment_uid, read_id)`. Tests must use a real pooled materialization and
duplicate read IDs across experiments.

### High: experiment latent has no transactional lifecycle or valid restart contract

The partitioned latent executor creates and mutates its final output directory,
writes each task, then the catalog, plots, spine, consolidated spine, and sidecar
in sequence (`tools/partitioned_latent.py:613-681`). It does not use the shared
stage lifecycle, a unique temporary publication directory, an artifact-complete
manifest, or configuration/source compatibility validation.

A failure can leave a partial store. More seriously, once `spine.h5ad` exists,
the CLI may skip a future run even if later publication steps failed or a task
store is missing. Force-redo also reuses final paths, so stale task groups are not
explicitly excluded from the new publication.

### High: latent limits read count, not estimated peak memory

Sequential units and chunked transforms are good controls. They do not bound the
fit matrix as a function of `n_reads * n_positions * dtype`, nor PCA/NMF/UMAP/CP
working allocations. A genome or wide-locus core can therefore exceed the
resource envelope even below `latent_max_fit_reads`. Plotting later reopens each
complete task result and also lacks a preflight.

The executor should resolve the shared resource envelope, estimate each enabled
algorithm, reduce the fit/chunk size when possible, and fail before allocation
when the minimum viable unit cannot fit. Persist the estimate, selected limits,
and observed peak in the task catalog/manifest.

### High: project embedding cache keys can collide across different data selections

The embedding hash includes feature/window/model parameters only
(`project/embedding_store.py:60-78`). `fit_or_extend_embedding` additionally
accepts canonical reference, set, modality, experiment filters, and stage, but
those inputs and source artifact identity are absent from the definition
(`project/embedding_store.py:216-287`). When a named set is used, its label also
does not incorporate the canonical reference.

This permits the same cache directory to represent different stage, modality,
experiment-filter, canonical-reference, or regenerated-source selections. A
matching molecule-name set can be treated as a cache hit even if feature values
changed in place. Artifact files are also written individually, so an interrupted
write can leave a mixed generation.

The definition needs the full semantic selection plus immutable source IDs or
content/config digests, schema and software versions. Publish a complete artifact
generation transactionally and validate its manifest before reading it.

### High: latent results are absent from molecule traceability

Project registration discovers read indexes only for preprocess, spatial, and HMM
(`project/registry.py:187-216`). The latent executor does not write a derived read
index. `ProjectCatalog.lookup_molecule` only scans `molecule_index` and catalog
keys ending in `_read_index` (`project/catalog.py:156-180`).

As a result, the Parquet-only trace path stops before latent. Locating a molecule's
latent record requires opening or scanning task stores. Latent should publish an
index containing at least experiment/molecule/read identity, reference/core,
group path, and group row.

### Medium: `from_adata_stage` conflicts with forced partitioned latent mode

Partitioned source discovery runs only when `from_adata_stage is None`
(`cli/latent_adata.py:174-180`). Setting both
`latent_execution_mode=partitioned` and `from_adata_stage` therefore leaves no
partitioned source and raises `FileNotFoundError`, even when the requested stage
has a valid spine. Either resolve the named stage to its partitioned spine or
reject the combination during configuration loading with a precise message.

### Medium: latent configuration accepts invalid ranges until runtime

Latent settings are converted to integers/floats but lack semantic validation.
Zero or negative component counts, iteration counts, neighbors, fit limits,
chunk sizes, and CP ranks should fail while loading configuration. Execution mode
should be validated there as well, not only after CLI setup.

### Medium: latent model provenance and deterministic extension are incomplete

Task outputs record result keys and fit-read counts but not fitted model artifacts,
fit-membership identities/digests, source/config hashes, or model library versions.
Seeded sampling is reproducible only while input ordering stays fixed. After an
experiment grows, the exact local model cannot be recovered and extended from
the published artifact; rerunning may define a new space.

Persist enough provenance to reproduce a task and decide whether an existing
model is compatible. If incremental experiment-local transformation is desired,
persist the fitted models and select fit reads by a stable identity-derived rule.

### Medium: registry refresh and documentation semantics can become stale

The latent executor regenerates the consolidated experiment spine, so a project
that already points to that file can observe new pointers through default
resolution. Its registry does not gain an explicit `latent` spine or latent index
until the experiment is re-added. Per-sample pointer inventories are likewise a
registration-time snapshot.

Additionally, current code prefers the consolidated experiment spine before the
documented HMM/spatial/preprocess/raw fallback (`project/registry.py:468-499`).
User documentation and CLI help should describe this actual behavior and explain
when `project add` is a refresh operation.

### Medium: project export provenance identifies the query, not a complete source generation

Partitioned export has strong transaction and resource behavior, but its manifest
does not provide immutable checksums/IDs for every resolved input spine, catalog,
task generation, and effective configuration. Recording those values would make
an export reproducible and allow consumers to reject a stale or mixed source set.

## Existing strengths to preserve

- Registry paths are portable and backward-compatible with older absolute-path
  registries.
- Canonical references are sequence-identity based rather than display-name based.
- Pooled identities use experiment-global molecule IDs and collision-free obs
  names.
- Project materialization performs a full-selection memory preflight before the
  first part is read.
- Project export keeps output partitioned and publishes through a temporary
  directory.
- Latent source selection prefers the most-derived compatible partitioned spine.
- Latent units reuse the shared analysis-core plan and preserve original-coordinate
  provenance.
- Latent coordinate scope is explicitly marked independent.
- Latent model fitting is capped, transforms are chunked, and units are sequential.
- The latent command remains standalone rather than silently increasing the full
  recipe's cost.

## Missing tests and acceptance criteria

### Project/latent boundary

- Register an experiment containing a real latent task store and prove the public
  API either returns a scoped latent result or raises an explicit unsupported-
  operation error. It must not silently return upstream-only data when latent
  representations were requested.
- Confirm two independently fitted cores/experiments are never concatenated into
  one latent coordinate matrix without an explicit alignment model.
- Verify the project-global embedding is labeled and persisted separately from
  experiment-local latent artifacts.

### Identity and traceability

- Join periodicity onto a real `project_adata` result.
- Use identical bare read IDs in two experiments and verify both results attach to
  the correct `molecule_uid`.
- Trace one molecule through raw, preprocess, HMM, and latent indexes without
  opening task Zarr stores.

### Lifecycle and resource behavior

- Inject failures after a latent task, after the catalog, after the spine, and
  during plots; no partial generation may be considered complete.
- Delete or corrupt a task store after success; the CLI must reject the completion
  marker and rebuild or report a precise error.
- Change an output-affecting latent setting or source artifact; restart validation
  must invalidate the old generation. Plot-only changes should not refit models.
- Exercise a wide core under a small resource envelope and confirm a pre-allocation
  adjustment or deterministic failure.
- Validate every latent numeric boundary at config load time.

### Cache provenance

- Prove project embeddings do not collide across stage, modality, experiment
  filter, canonical reference, or source generation.
- Change feature values while retaining molecule IDs; the old embedding must not
  be treated as a valid cache hit.
- Interrupt artifact publication; readers must continue to see the prior complete
  generation or no generation, never a mixture.

## Dependency-ordered remediation backlog

1. **Fix project analysis identity joins.** Move periodicity cache schema and joins
   to `molecule_uid` (or the explicit experiment/read composite), migrate or reject
   old cache schema, and add collision tests.
2. **Give latent the shared publication contract.** Add stage lifecycle state,
   source/config compatibility checks, a completed artifact manifest, unique temp
   output, atomic publication, and failure-injection tests.
3. **Add latent resource planning and config validation.** Estimate enabled model
   allocations, derive limits from the resource envelope, validate all ranges, and
   persist planned/observed resource data.
4. **Publish a latent derived-read index.** Include project-global molecule identity,
   task scope, group path, and group row; teach registry discovery and molecule
   lookup about it.
5. **Define a scoped latent read API.** Read task-local `obsm`/`varm`/labels by
   molecule and core. Make project CLI behavior explicit: either expose scoped
   artifacts or reject latent fields in generic genomic materialization/export.
6. **Harden the project embedding store.** Expand semantic/source provenance,
   version its schema, validate reads, and publish each generation transactionally.
7. **Persist latent model provenance if extension is a requirement.** Store models,
   stable fit membership, dependency versions, and source/config digests; otherwise
   document latent results as non-extendable derived artifacts.
8. **Align CLI help and documentation.** Document consolidated-spine preference,
   project refresh behavior, experiment-local versus project-global spaces, and
   the supported `from_adata_stage` combinations.

These items should remain separate, focused PRs. Items 1-4 establish identity,
publication, resource, and indexing foundations before a project-facing latent
reader is introduced.

## Completion definition

The package can claim project/latent compatibility when:

1. selecting latent at project scope cannot silently omit requested latent data;
2. experiment-local coordinate scope is preserved and never conflated with the
   project-global embedding space;
3. every latent molecule is addressable through the project identity/index
   contract;
4. project analyses and caches use collision-free identities and complete source
   provenance; and
5. latent execution obeys the same validated, resource-aware, restart-safe
   publication contract as the rest of the partitioned pipeline.

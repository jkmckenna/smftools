# Semantic DAG and variant-preprocessing implementation plan

**Plan date:** 2026-07-27

**Repository:** `smftools`

**Program status:** Complete — SDV-01 through SDV-14 merged

**Repository state at planning time:** `293ec85`
(`feature/project-latent-acceptance`)

**Source audit:**
[variant_preprocessing_incremental_reprocessing_audit.md](variant_preprocessing_incremental_reprocessing_audit.md)

**Companion implementation plan:**
[project_and_latent_partitioned_pipeline_implementation_plan.md](project_and_latent_partitioned_pipeline_implementation_plan.md),
with PL-15 through PL-23 completed at the repository state reviewed

## Objective

Incrementally introduce an engine-neutral semantic dependency graph throughout
the smftools experiment/project analysis surface and absorb the useful
standalone variant analyses into partitioned preprocessing before QC filtering
and duplicate removal.

The completed program must:

- preserve all current supported CLI entry points while migrating their
  orchestration behind one semantic planner;
- distinguish scientific dependency/compatibility from process scheduling;
- retain raw alignment and alignment-rescue ownership in the raw stage;
- calculate variant evidence for every available raw molecule before QC masks
  exclude reads;
- support reporting-only variant analysis before enabling hard filtering;
- publish explicit variant, non-variant, combined-QC, and dedup masks;
- publish variant-resolved chimera metrics before and after duplicate removal;
- apply newly introduced compatible analyses to prior partitioned experiments
  without mutating their published generations;
- invalidate dedup, downstream experiment stages, and project products when
  their actual consumed scientific inputs change;
- reuse the identity, lifecycle, resource, index, and project-provenance
  contracts completed in PL-15 through PL-23;
- remain callable from ordinary Python, Click, Snakemake, Nextflow, Slurm
  arrays, or another external executor; and
- establish the package/container contracts needed for a later independent
  Nextflow and possible nf-core pipeline.

This is an incremental feature program. Each implementation item below is
intended to be a focused branch and PR. Feature branches should be cut from the
then-current `main`, should not bump `src/smftools/_version.py`, and should not
silently change default read filtering.

## Current baseline

The plan assumes the following implemented behavior at `293ec85`:

- `smftools experiment full` runs raw, preprocess, spatial, HMM, and latent by
  default, with latent configurable as an opt-out.
- Partitioned experiment stages publish completion records in
  `experiment_manifest.json`.
- Latent and project embeddings have immutable generation layouts,
  transactional current pointers, source/config compatibility, and
  failure-injection coverage.
- Experiment/project identity uses `experiment_uid` and `molecule_uid`.
- Partitioned preprocessing retains every raw spine row and represents QC and
  dedup as masks rather than deleting raw records.
- Resource envelopes, task catalogs, relative artifact pointers, relocation
  validation, and derived read indexes are already established patterns.
- The standalone `variant` command is legacy-monolithic, post-QC/post-dedup,
  pair-specific, and absent from the standard partitioned full workflow.
- Preprocess completion is still stage-granular and its canonical output is not
  an immutable analysis generation with per-node reuse.

The completed PL-15 through PL-23 implementations are dependencies and design
examples. This program must not reopen their scientific contracts or create
parallel identity, current-pointer, model, or cache authorities.

## Program finding IDs

These IDs provide stable references for PR descriptions and acceptance tests.

| ID | Severity | Finding |
|---|---|---|
| SDV-C1 | Critical | Standalone variant analysis cannot run against the partitioned preprocess pipeline |
| SDV-C2 | Critical | Current variant execution occurs too late to annotate all reads, drive QC, or measure pre-dedup events |
| SDV-H1 | High | Experiment/project commands do not share an analysis-level dependency and compatibility planner |
| SDV-H2 | High | Preprocess publication and restart are stage-granular rather than immutable and per-analysis |
| SDV-H3 | High | Existing variant completion flags do not identify reference set, parameters, algorithm, source generation, or outputs |
| SDV-H4 | High | Variant “multi-reference” behavior assumes exactly one reference pair and global completion flags |
| SDV-H5 | High | Existing chimera labels are not sufficiently specific or evidence-gated for default hard filtering |
| SDV-H6 | High | Variant-aware QC/dedup changes are not connected to downstream experiment/project invalidation |
| SDV-M1 | Medium | Population mismatch summaries and deterministic known-reference calls have mixed cohort/provenance semantics |
| SDV-M2 | Medium | Variant-derived pre/post-dedup metrics lack explicit durable cohort denominators |
| SDV-M3 | Medium | External workflow callers lack uniform result, validation, versions, and immutable-output contracts |
| SDV-M4 | Medium | There is no production smftools container or external workflow acceptance profile |

## Agreed design contracts

These contracts constrain the PRs below. Changing one requires a design review
and an update to this plan before implementation proceeds.

### The internal DAG is semantic, not an execution engine

The internal graph owns:

- scientific dependencies;
- semantic configuration selection;
- algorithm and output-schema compatibility;
- input/dependency artifact identity;
- invalidation;
- plan explanations;
- artifact validation; and
- immutable generation composition.

It does not own:

- HPC/cloud submission;
- container selection;
- retries;
- external file staging;
- cross-experiment distributed scheduling; or
- Nextflow/Snakemake cache behavior.

Existing bounded task planners, process pools, resource envelopes, reducers, and
task catalogs remain the execution layer beneath a semantic node. Nextflow,
Snakemake, or another external engine may later replace that physical execution
layer only through a public plan/run-task/finalize protocol.

### Use one node model across three graph scopes

The same node contract is used at three scopes:

1. **Experiment stages:** raw -> preprocess -> spatial -> HMM -> latent.
2. **Within-stage analyses:** variant evidence, QC masks, duplicate clustering,
   metrics, and plots.
3. **Project analyses:** frozen experiment-source selection, genomic
   materialization, sample analysis, and project-global embeddings.

The graphs may be planned separately, but they must use the same compatibility
vocabulary and result-state model.

Physical reference/core/barcode/read chunks are not first-class semantic graph
nodes. They remain rows in a versioned task catalog owned by one semantic node.
Changing worker count or physical chunking must not silently redefine
scientific identity.

### CLI commands select targets; they do not encode dependency order

Existing Click commands become thin target selectors:

```text
experiment raw         -> experiment.raw.complete
experiment preprocess  -> experiment.preprocess.complete
experiment spatial     -> experiment.spatial.complete
experiment hmm         -> experiment.hmm.complete
experiment latent      -> experiment.latent.complete
experiment full        -> final enabled experiment target
experiment variant     -> preprocess variant/report targets (compatibility alias)
```

The `full` implementation must not maintain a second hand-written dependency
sequence once the experiment graph is authoritative.

Not every CLI action should become a persisted analysis node:

- `experiment load` may remain an optional dense-cache materialization node or
  utility because it does not define a new scientific source stage.
- `experiment batch` remains an outer iterator over independent experiment
  target requests; external engines should invoke individual experiments.
- concatenate, FASTQ export, current plotting, project listing, and similar
  queries/actions consume validated graph outputs but do not become upstream
  scientific dependencies.
- project `init`, `add`, and `remove` mutate the project registry; the resulting
  registry/source snapshot is an input identity, not an analysis artifact that
  hides those mutations.

### Planning is read-only and explainable

A plan classifies each requested node as one of:

- `compatible`;
- `missing`;
- `stale_config`;
- `stale_algorithm`;
- `stale_input`;
- `invalid_artifact`;
- `dependent_recompute`; or
- `blocked_missing_input`.

A planning call performs no writes other than optional ordinary logs. A
machine-readable plan includes the requested target, topological order, current
generation, selected/rejected reuse, exact invalidation reason, and expected
output types.

### Compatibility is node- and channel-specific

Each semantic node declares:

- stable `analysis_id`;
- graph scope;
- algorithm version;
- output schema version;
- exact semantic configuration keys;
- required input channels and schemas;
- upstream node/channel dependencies;
- logical task scope;
- produced artifacts and channels;
- validation rules; and
- downstream invalidation edges.

The compatibility key is equivalent to:

```text
analysis_id
+ algorithm_version
+ output_schema_version
+ semantic_config_hash
+ ordered input artifact identities/checksums
+ dependency result IDs/channel fingerprints
+ logical scope identity
```

Package, Python, dependency, git, machine, resource, and timing information is
recorded as provenance. A package upgrade alone does not invalidate every node;
scientific changes increment that node's algorithm/schema version.

Downstream nodes depend on declared channels rather than an undifferentiated
whole-stage timestamp. A newly added diagnostic plot should not eventually
invalidate latent compute. During initial migration, conservative whole-source
invalidation is acceptable where channel fingerprints do not yet exist, but it
must be explicit rather than silently ignored.

### Published generations are immutable

The experiment manifest remains authoritative for stage completion. An
immutable generation manifest records the analysis-node results composing that
stage generation.

Adding a missing analysis means:

1. validate the current source generation;
2. plan compatible, missing, stale, and blocked nodes;
3. build a unique `.staging/<generation_id>` tree;
4. reuse validated immutable artifacts or compute replacements;
5. validate the complete staged generation;
6. atomically publish `generations/<generation_id>`;
7. atomically advance `current.json`;
8. update the stage completion record; and
9. refresh the consolidated experiment spine and project discovery only after
   publication.

No command may append files beneath the current published generation in place.
A failure leaves the prior complete generation current and readable.

### Raw owns alignment facts; preprocess owns variant interpretation

Raw continues to own:

- basecalling/demultiplexing/alignment invocation where configured;
- primary/secondary alignment rescue;
- final raw `Reference_strand`, CIGAR, start, and mapped-coordinate facts;
- sequence, mismatch, base-quality, read-span, and modification-signal
  extraction; and
- immutable raw molecule identity.

Preprocess owns:

- reference-to-reference alignment for variant annotation;
- informative reference-site catalogs;
- per-read known-reference evidence and segmentation;
- cohort-dependent mismatch frequencies;
- variant QC masks and reasons;
- duplicate interaction;
- pre/post-dedup variant metrics; and
- variant plots.

Variant analysis must not retroactively change raw reference assignment without
a separately designed raw-stage algorithm/version.

### Variant evidence precedes filtering

Variant evidence is computed for every raw molecule for which required sequence
channels are available. It must complete before QC decisions or dedup candidate
selection exclude reads.

Raw may already contain primitive read/mapping/chimera metrics. “Before QC”
means before applying the preprocess masks, not that those immutable primitive
metrics must be recalculated later.

The target mask contract is:

```text
passes_read_qc
passes_modification_qc
passes_variant_qc
passes_nonvariant_qc = passes_read_qc & passes_modification_qc
passes_qc = passes_nonvariant_qc & passes_variant_qc
passes_dedup = passes_qc & ~is_duplicate
```

Failure reasons remain queryable even when a final mask is false.

### Variant reporting lands before hard filtering

The first integrated behavior is reporting-only. Variant evidence, cohort
metrics, and plots are produced, but `passes_variant_qc` is true for every row.

Hard filtering requires a later explicit configuration mode and validated
minimum evidence/callability policies. The broad
`chimeric_variant_sites` diagnostic and the transition-based breakpoint flag
remain distinct. Neither becomes a default hard filter merely because it
already exists.

### Known-reference evidence and cohort mismatch frequencies are separate nodes

Reference-to-reference informative sites are deterministic given reference
sequences and calling policy. Per-read calls depend on that catalog and raw read
evidence.

Population mismatch frequencies depend on an explicitly named cohort. They do
not define known-reference informative sites, and a frequency learned from a
cohort must not filter the same cohort without an explicit, tested policy.

### Variant identity is reference-set aware

Persistent per-molecule variant results use:

```text
(experiment_uid, molecule_uid, variant_reference_set_id,
 analysis_generation_id)
```

Position overlays additionally identify their position/core owner. Exactly one
logical owner produces the final per-molecule classification across tile
boundaries.

The stable reference-set ID includes canonical reference identities, sequence
checksums, orientation, alignment/scoring semantics, conversion semantics, and
informative-site policy. A display label alone is not identity.

The initial implementation may provide parity for two references, but its
persistent schemas and completion flags must not assume there can only ever be
one pair.

### Project dependencies use frozen source snapshots

Project analysis nodes consume an explicit snapshot of:

- active experiment IDs;
- experiment and molecule identity;
- selected canonical references/sets;
- stage and generation IDs;
- required channel fingerprints;
- source artifact identities; and
- selection/model configuration.

A reporting-only variant artifact should not eventually invalidate a project
embedding that does not consume it. A variant-QC or dedup change that alters
membership/features of existing molecules follows PL-21's full-refit behavior;
it is not compatible-growth reuse.

Experiment-local latent spaces remain independent and are never pooled by this
graph. Project-global embeddings remain a separately fitted product.

### External workflow compatibility remains engine-neutral

No core node imports Nextflow or Snakemake. Workflow-facing commands use
read-only inputs, explicit output roots, nonzero failure signaling, stable JSON
results, validation commands, and version artifacts.

The first external workflow should invoke one isolated
`smftools experiment full` process per experiment from an aligned,
modification-bearing BAM. Fine-grained stage or partition scatter is deferred
until commands accept immutable upstream bundles and publish self-contained
outputs.

## Target semantic graph

The initial experiment graph should converge on:

```text
experiment.raw.store
  |
  +--> preprocess.variant.reference_catalog
  |       |
  |       +--> preprocess.variant.read_evidence
  |
  +--> preprocess.read_qc
  |
  +--> preprocess.derived_layers
          |
          +--> preprocess.modification_metrics
                    |
                    +--> preprocess.modification_qc

preprocess.read_qc + preprocess.modification_qc
  -> preprocess.nonvariant_qc

preprocess.variant.read_evidence
  -> preprocess.variant.qc

preprocess.nonvariant_qc + preprocess.variant.qc
  -> preprocess.combined_qc
  -> preprocess.duplicate_clusters

variant evidence + named QC cohorts + duplicate clusters
  -> preprocess.variant.cohort_metrics
  -> preprocess.variant.plots

all required preprocess nodes
  -> experiment.preprocess.complete
  -> experiment.spatial.complete
  -> experiment.hmm.complete
  -> experiment.latent.complete
```

Population mismatch-frequency nodes may consume `all_aligned`,
`pre_dedup_nonvariant_qc`, or another explicitly configured cohort. Their
cohort identity must be present in their result key and output table.

The project graph should converge on:

```text
project.registry.snapshot
  + experiment generation/channel snapshots
    -> project.genomic_selection
      -> project.materialization/export
      -> project.sample_analysis
      -> project.embedding.feature_matrix
        -> project.embedding.generation
```

Registry management and read-only query/export actions may use the planner
without becoming immutable scientific nodes themselves.

## Semantic-node and manifest schemas

The initial node specification should be intentionally small:

```text
analysis_id
scope
dependencies
consumed_channels
produced_channels
semantic_config_keys
algorithm_version
output_schema_version
task_scope
validator_id
```

Executors and validators are registered code callables, not serialized in
manifests.

Each published node result records at least:

```text
analysis_id
state
result_id
algorithm_version
output_schema_version
semantic_config_hash
input_artifact_ids
dependency_results and consumed channel fingerprints
logical_task_plan_digest
artifacts with relative paths/checksums/schema/kind
reused_from_generation_id, when applicable
started_at/completed_at
execution provenance
```

The graph definition version and node-result schema version are independent.
Changing graph wiring does not automatically change an unchanged node's
scientific result identity.

## Preprocess generation layout

A concrete initial layout is:

```text
preprocess_adata_outputs/
  current.json
  generations/<preprocess_generation_id>/
    generation_manifest.json
    spine.h5ad
    task_catalog.parquet
    read_index/
    analyses/
      preprocess.read_qc/<result_id>/...
      preprocess.derived_layers/<result_id>/...
      preprocess.modification_qc/<result_id>/...
      preprocess.variant.reference_catalog/<result_id>/...
      preprocess.variant.read_evidence/<result_id>/...
      preprocess.variant.qc/<result_id>/...
      preprocess.duplicate_clusters/<result_id>/...
      preprocess.variant.cohort_metrics/<result_id>/...
    plots/
      plot_catalog.parquet
```

The canonical preprocess spine is a validated view/pointer for the current
generation. Task writers never use it as their working directory.

Artifact reuse must be relocation-safe. Copying is always valid. Hard links or
content-addressed references are allowed only when retention ownership and
relocation behavior are explicitly validated.

## Configuration and migration contract

### Normalize legacy configuration before adding new syntax

The current
`references_to_align_for_variant_annotation=[seq1_col, seq2_col]` setting
should be normalized into an in-memory `VariantReferenceSet` rather than
consumed directly by new executors.

The first config PR should:

- preserve the legacy two-element field as a compatibility input;
- validate that both or neither values are supplied;
- resolve referenced sequences against the raw reference catalog;
- reject ambiguous aliases;
- assign a deterministic reference-set ID; and
- warn only when a deprecated spelling is actually used.

A structured multi-reference input may later use a dedicated path to a
versioned CSV/JSON/YAML schema. It should not overload the existing two-element
CSV cell with an unversioned nested language.

### Use one mode for analysis/filter intent

A preferred public contract is:

```text
variant_analysis_mode = off | report | filter
```

- `off`: no integrated variant work is requested.
- `report`: compute evidence, metrics, and plots; all rows pass variant QC.
- `filter`: apply validated variant QC thresholds before dedup.

The default for the first integration release should preserve current
preprocess filtering behavior. The exact transition from explicit opt-in to
automatic reporting when a reference set is configured requires a user-facing
migration decision before SDV-07.

Legacy `omit_chimeric_reads` and overlay settings must be mapped to explicit
consumer behavior rather than silently reinterpreted as the new variant-QC
policy.

### Force flags become planner requests

New work should not add another collection of per-function
`force_redo_*` booleans. The planner should support:

- force a target node and its dependents;
- recompute all nodes in a stage generation; and
- regenerate plots while reusing compatible compute.

Existing flags remain supported through a normalization layer until separately
deprecated. A force request never permits in-place mutation of a current
generation.

## Delivery strategy

Use one focused branch/PR per item. The primary dependency chain is:

```text
SDV-01 semantic node model
  -> SDV-02 experiment graph/plan integration
  -> SDV-03 immutable preprocess generations
  -> SDV-04 analysis-level reuse and upgrade planning

SDV-05 variant semantics/reference-set contract
  + SDV-03 immutable preprocess generations
    -> SDV-06 partitioned per-read variant evidence

SDV-03 + SDV-04 + SDV-06
  -> SDV-07 reporting-only preprocess integration
  -> SDV-08 cohort metrics and plots
  -> SDV-09 variant QC and dedup policy

SDV-02 + SDV-04
  -> SDV-10 project graph and dependency invalidation

SDV-07 + SDV-08 + SDV-09 + SDV-10
  -> SDV-11 standalone variant compatibility migration

SDV-02 + SDV-10
  -> SDV-12 workflow-facing CLI/artifact contract
  -> SDV-13 production CPU container

all core implementation PRs
  -> SDV-14 documentation and integrated acceptance

SDV-12 + SDV-13 + reporting-only variant integration
  -> NF-01 independent Nextflow prototype
  -> NF-02 optional nf-core proposal/hardening
```

SDV-01 and SDV-05 can proceed independently. SDV-10 may begin before variant
filtering is enabled, but its final acceptance must cover a variant-QC source
change. SDV-12/SDV-13 do not need to wait for hard variant filtering if the
stable reporting artifacts are already available.

### Backward-compatible rollout checkpoints

Every checkpoint must be releasable without unfinished later work:

1. **After SDV-02:** planning and full orchestration use the experiment graph,
   but scientific outputs and stage executors remain unchanged.
2. **After SDV-04:** preprocess outputs use immutable generations and can reuse
   existing analysis nodes, but variant behavior remains unchanged.
3. **After SDV-06:** partitioned variant evidence can be produced and validated
   through internal APIs/tests, but it does not affect normal preprocessing.
4. **After SDV-07:** enabled variant analysis runs in reporting mode through
   preprocess/full, with identical QC and dedup membership to the prior
   behavior.
5. **After SDV-08:** pre/post-dedup variant metrics and plots are public durable
   outputs, still without automatic filtering.
6. **After SDV-09:** filter mode is available only through explicit validated
   configuration; report/off compatibility remains.
7. **After SDV-11:** the standalone command is a compatibility alias, with
   legacy artifact readers retained according to the migration policy.
8. **After SDV-13:** a workflow caller can use the pinned CPU image, but the
   external Nextflow repository remains optional and independently versioned.

Each PR must update persistent schema versions and migration readers in the same
change that writes the new schema. A producer must not merge before its reader
and validator. Performance-sensitive PRs must include a bounded-memory test or
benchmark fixture and must not require full all-molecule dense materialization.

## Ordered core PR backlog

| ID | Suggested branch | Primary outcome | Finding coverage | Depends on |
|---|---|---|---|---|
| SDV-01 | `feature/semantic-analysis-graph` | Engine-neutral node registry, compatibility model, and read-only planner | SDV-H1, SDV-H3 | PL-15–PL-23 |
| SDV-02 | `feature/experiment-semantic-planning` | Experiment commands select graph targets; add explainable experiment plan output | SDV-H1 | SDV-01 |
| SDV-03 | `feature/preprocess-immutable-generations` | Transactional immutable preprocess publication using PL-17 conventions | SDV-H2 | SDV-01, SDV-02 |
| SDV-04 | `feature/preprocess-incremental-upgrades` | Per-analysis compatibility/reuse and read-only upgrade planning | SDV-H2, SDV-H3, SDV-H6 | SDV-03 |
| SDV-05 | `feature/variant-reference-contract` | Explicit reference-set semantics, stable identity, pure parity-tested kernels | SDV-H4, SDV-H5, SDV-M1 | PL-15, PL-16, PL-22 |
| SDV-06 | `feature/partitioned-variant-evidence` | Partitioned-native all-molecule variant evidence and indexed artifacts | SDV-C1, SDV-C2, SDV-H3, SDV-H4 | SDV-03, SDV-05, PL-18, PL-19 |
| SDV-07 | `feature/preprocess-variant-reporting` | Variant evidence runs before filtering in preprocess, reporting-only by default | SDV-C1, SDV-C2, SDV-H5 | SDV-03, SDV-04, SDV-06 |
| SDV-08 | `feature/variant-cohort-metrics` | Durable named-cohort pre/post-dedup metrics and preprocess plots | SDV-M1, SDV-M2 | SDV-07 |
| SDV-09 | `feature/variant-qc-dedup-policy` | Configurable evidence-gated variant QC and explicit duplicate interaction | SDV-H5, SDV-H6, SDV-M2 | SDV-08 |
| SDV-10 | `feature/project-semantic-planning` | Project source snapshots, graph planning, and channel-sensitive invalidation | SDV-H1, SDV-H6 | SDV-02, SDV-04, PL-20, PL-21 |
| SDV-11 | `fix/variant-command-preprocess-alias` | Backward-compatible variant command migration and legacy-output policy | SDV-C1, SDV-H3, SDV-H4 | SDV-07 through SDV-10 |
| SDV-12 | `feature/workflow-cli-contract` | Stable result JSON, validation, versions, resource, and output-root contracts | SDV-M3 | SDV-02, SDV-10 |
| SDV-13 | `feature/production-cpu-container` | Pinned BAM-entry production image and container smoke tests | SDV-M4 | SDV-12 |
| SDV-14 | `feature/semantic-variant-acceptance` | Integrated acceptance, migration guidance, CLI/docs completion | All | SDV-01 through SDV-13 |

## SDV-01 — semantic node model and planner

### Scope

- Add a small engine-neutral semantic graph package.
- Define immutable typed records for node specifications, dependencies,
  produced/consumed channels, compatibility inputs, plan decisions, and node
  results.
- Validate unique node IDs, known dependencies, acyclic graphs, and legal
  scope transitions.
- Implement deterministic topological planning for requested targets.
- Implement compatibility-key construction from semantic inputs only.
- Implement read-only classification and reason reporting.
- Support an in-memory registry assembled explicitly by experiment/project
  graph builders; avoid import-time global side effects.
- Provide a stable JSON-serializable plan schema.
- Add test-only fake executors/validators, but do not change a production stage
  execution path in this PR.

### Suggested primary files

- new `src/smftools/pipeline/semantic_graph.py`
- new `src/smftools/pipeline/analysis_registry.py`
- new `src/smftools/pipeline/compatibility.py`
- `src/smftools/constants.py` for frequently reused schema-version constants
- new focused unit tests under `tests/unit/pipeline/`

The exact new package name may be adjusted before implementation, but it should
describe internal pipeline semantics rather than imply Nextflow/Snakemake
execution.

### Required tests

- Registration order does not change topological plan order.
- Duplicate node IDs, cycles, unknown dependencies, and illegal cross-scope
  dependencies fail with actionable messages.
- Changing an unrelated config field does not invalidate a node.
- Changing a declared semantic config field invalidates the node and declared
  dependents.
- Changing algorithm or output-schema version invalidates the correct node.
- Changing an input artifact or dependency result invalidates the correct node.
- Worker count, task order, timing, and machine provenance do not alter the
  compatibility key.
- Planning performs no artifact or current-pointer writes.
- Plan JSON is deterministic for identical logical inputs.

### Exit gate

A pure unit-tested planner can explain compatibility and dependency-driven
recomputation without importing Click, executing a stage, or mutating a run.

## SDV-02 — experiment graph and CLI planning

### Scope

- Define coarse experiment nodes for raw, preprocess, spatial, HMM, and latent.
- Represent configured opt-outs, including `full_run_latent`, as target
  resolution rather than hidden edges.
- Adapt current stage config hashes and input artifact IDs into node
  compatibility inputs without changing their executors.
- Add a read-only experiment plan command or equivalent option with human and
  JSON output.
- Make `experiment full` execute the resolved topological plan.
- Preserve direct stage commands as target aliases.
- Ensure stage wrappers remain independently testable outside Click.
- Keep `load`, `batch`, export, concatenate, and plotting behavior explicit as
  utility/consumer commands rather than forcing every command into the
  scientific graph.
- Register the legacy standalone variant/chimeric stages only as compatibility
  leaves at this point; do not yet change their execution.

### Suggested primary files

- `src/smftools/cli/recipes.py`
- `src/smftools/cli_entry.py`
- `src/smftools/cli/helpers.py`
- new `src/smftools/pipeline/experiment_graph.py`
- `src/smftools/informatics/experiment_manifest.py`
- CLI and planner tests

Before editing CLI code, re-read `src/smftools/cli/AGENTS.md`.

### Required tests

- Full resolves raw -> preprocess -> spatial -> HMM -> latent by default.
- Latent opt-out resolves HMM as the final target.
- Requesting HMM plans only missing/incompatible dependencies through HMM.
- A compatible stage is skipped for the same reason as the current lifecycle.
- A stale source or config produces an explicit plan reason.
- Direct stage commands and full produce equivalent stage target requests.
- Planning JSON is valid on fresh, partially complete, and complete runs.
- `batch` still reports per-experiment failures and does not hide target
  planning errors.

### Exit gate

Experiment dependency order has one authoritative graph, while current stage
executors and public direct-stage commands remain behaviorally compatible.

## SDV-03 — immutable preprocess generations

### Scope

- Introduce preprocess `.staging`, `generations`, and atomic `current.json`
  publication using the completed latent lifecycle conventions.
- Add a versioned preprocess generation manifest.
- Move the current preprocess spine, sidecars, task catalog, derived read index,
  stores, and plot catalog under the generation root.
- Keep a canonical current preprocess spine/path for compatibility.
- Validate all required files, schemas, task counts, checksums, and relative
  pointers before advancing current.
- Record the raw source artifact/generation identity.
- Refresh the consolidated experiment spine only after successful publication.
- Initially allow the whole current preprocess computation to rerun; per-node
  reuse is SDV-04.
- Preserve the current published generation after an injected failure.
- Provide relocation-safe discovery to the project registry.

### Suggested primary files

- `src/smftools/cli/preprocess_adata.py`
- `src/smftools/preprocessing/partitioned_executor.py`
- `src/smftools/informatics/experiment_manifest.py`
- `src/smftools/informatics/experiment_spine.py`
- `src/smftools/informatics/partition_read.py`
- `src/smftools/project/registry.py`
- a focused preprocess generation-store module
- partitioned preprocess lifecycle/failure tests

Prefer sharing small proven lifecycle utilities with latent rather than
rewriting the completed latent implementation in this PR.

### Required failure-injection and migration tests

- Failure before any staged artifact publishes no current generation.
- Failure after task writes but before validation preserves the prior current
  generation.
- Failure after generation validation but before current-pointer update leaves
  readers on the prior generation.
- A malformed current pointer or checksum mismatch is rejected.
- The canonical spine never points at a partial pre-dedup output.
- Existing canonical preprocess output can be recognized as legacy schema and
  either imported conservatively or recomputed from raw.
- Moving the complete experiment tree preserves generation resolution.
- Project re-add/refresh discovers the published preprocess generation.

### Exit gate

Preprocess completion is generation- and manifest-driven; no partial or failed
run can become current merely because `spine.h5ad` exists.

## SDV-04 — analysis-level compatibility and incremental upgrade

### Scope

- Register the existing partitioned preprocess operations as semantic nodes,
  initially at useful reducer/task-family granularity.
- Add node-result records to the preprocess generation manifest.
- Add per-node output validation and result/channel fingerprints.
- Add a read-only preprocess upgrade plan against an existing partitioned raw
  or preprocess generation.
- Build a new generation by reusing validated compatible node artifacts and
  computing missing/stale nodes.
- Record `reused_from_generation_id` for reused results.
- Define force-target behavior through planner requests.
- Define conservative fallback when a legacy generation lacks sufficient node
  provenance.
- Preserve prior generations; cleanup/retention is explicitly out of scope.
- Make source changes, corrupt artifacts, and missing raw channels produce
  distinct planner outcomes.

### Suggested primary files

- the semantic graph package from SDV-01
- preprocess generation-store module from SDV-03
- `src/smftools/preprocessing/partitioned_executor.py`
- `src/smftools/preprocessing/dispatch_plan.py`
- `src/smftools/informatics/sidecar_manifest.py`
- `src/smftools/cli/preprocess_adata.py`
- upgrade/reuse/fault-injection tests

### Required tests

- A new independent node is added without rerunning compatible task nodes.
- An algorithm-version change reruns that node and its dependents.
- A semantic config change reruns only affected nodes/dependents.
- An unrelated downstream/plot config change does not invalidate preprocess
  compute.
- A source raw/reference change cannot be a cache hit.
- A corrupt reused artifact prevents publication.
- Plot-only regeneration reuses compute.
- The old generation remains current after failed upgrade.
- A legacy H5AD or generation with insufficient provenance reports its limits
  rather than guessing compatibility.

### Exit gate

A prior partitioned raw/preprocess result can receive a new compatible analysis
through a newly published preprocess generation without modifying the old
generation or rerunning unrelated validated nodes.

## SDV-05 — variant reference-set and scientific semantics

### Scope

- Define and validate an in-memory `VariantReferenceSet` contract.
- Normalize the legacy two-column reference configuration into that contract.
- Calculate stable reference-set identity from canonical sequences and
  semantics.
- Separate pure reference alignment/informative-site calculation from AnnData
  mutation and plotting.
- Separate pure per-read call/segment calculation from legacy column naming.
- Explicitly define:
  - substitution call semantics;
  - insertion/deletion support or intentional exclusion;
  - no-call and uninformative states;
  - callability and evidence counts;
  - transition/breakpoint chimera semantics;
  - broad other-reference segment semantics; and
  - conversion-modality reference normalization.
- Preserve a compatibility adapter for existing variant functions.
- Add parity fixtures covering current known two-reference behavior.
- Decide the structured multi-reference schema, but do not require a
  multi-allelic caller in this PR unless separately scoped.

### Suggested primary files

- `src/smftools/config/experiment_config.py`
- `src/smftools/config/default.yaml`
- `src/smftools/preprocessing/append_sequence_mismatch_annotations.py`
- `src/smftools/preprocessing/append_variant_call_layer.py`
- new focused reference-set/pure-kernel modules
- existing and new variant unit fixtures

### Required tests

- Legacy two-reference config normalizes deterministically.
- Supplying one member, ambiguous members, or missing sequence sources fails
  before execution.
- Relocating inputs does not change the reference-set ID.
- Changing sequence, orientation, scoring, conversion, or informative-site
  policy changes the ID.
- Legacy fixtures retain expected substitution calls and segment labels.
- Sparse, no-call, fully discordant, one-breakpoint, edge, middle, and
  multi-segment examples have explicit expected results.
- Existing broad and transition-based chimera concepts cannot be confused by
  output names or schemas.

### Exit gate

Variant scientific semantics are pure, explicitly versioned, reference-set
aware, and testable without a monolithic AnnData or plotting stack.

## SDV-06 — partitioned-native variant evidence

### Scope

- Plan variant work from the raw spine/reference catalog.
- Calculate reference-set informative-site catalogs once per experiment/set.
- Calculate per-read evidence from raw ragged sequence/mismatch/read-span
  channels for all available molecules.
- Use PL-18 resource estimates and bounded dispatch.
- Guarantee one logical owner for each final per-molecule result across
  position-core boundaries.
- Store:
  - reference-set/informative-site catalog;
  - molecule-keyed variant `obs` sidecar;
  - sparse informative calls and segment/breakpoint events;
  - optional bounded positional overlay artifacts;
  - versioned task catalog; and
  - a relocation-safe variant read/event index.
- Key results by `experiment_uid`, `molecule_uid`, and
  `variant_reference_set_id`.
- Register outputs in sidecar and generation manifests.
- Do not yet change `passes_qc` or dedup.

### Suggested primary files

- `src/smftools/preprocessing/dispatch_plan.py` or a focused variant planner
- `src/smftools/preprocessing/partitioned_executor.py` or a focused variant
  executor
- `src/smftools/informatics/ragged_store.py`
- `src/smftools/informatics/partition_read.py`
- `src/smftools/informatics/derived_read_index.py` or a separate versioned
  variant index
- variant partition/index tests

Do not insert variant records into the PL-19 latent index or reuse latent
coordinate-owner semantics.

### Required tests

- Every raw molecule with required channels receives one final evidence record.
- Results are invariant to worker count, task order, memory envelope, and
  physical chunking.
- A read crossing adjacent position cores is finalized once.
- Breakpoints crossing a core boundary are detected.
- Two reference sets do not collide in artifacts or indexes.
- Duplicate bare read IDs across experiments remain distinct.
- Querying selected molecules/reference sets does not open unrelated task
  stores.
- Relocated experiment outputs remain readable.
- Missing required raw channels produce `blocked_missing_input`.

### Exit gate

Partitioned raw data can produce validated, indexed, all-molecule variant
evidence without constructing or writing a legacy monolithic variant H5AD.

## SDV-07 — reporting-only variant integration in preprocessing

### Scope

- Add reference catalog and read-evidence nodes to the preprocess graph before
  mask application and duplicate selection.
- Add `passes_variant_qc`, initially true for every molecule in report mode.
- Add `passes_nonvariant_qc` and retain the current final behavior through
  `passes_qc`.
- Preserve explicit QC reason columns.
- Publish variant results through the preprocess spine/sidecars and normal
  materialization/query paths.
- Add the normalized `variant_analysis_mode` configuration.
- Ensure `experiment full` receives variant outputs naturally through the
  preprocess target when enabled.
- Make current downstream overlays consume authoritative preprocess outputs
  rather than a best-effort standalone variant H5AD.
- Keep hard filtering disabled in this PR.

### Required behavior decision before implementation

Choose one migration default:

1. `variant_analysis_mode=off` unless explicitly enabled; or
2. automatically select `report` when a valid legacy/new reference set is
   configured.

Either choice must preserve QC/dedup membership and be documented. No first
release may silently select `filter`.

Decision: use option 2. `variant_analysis_mode=auto` resolves to `report` when
a valid two-member reference pair is configured and otherwise resolves to
`off`. Explicit `off` remains available, and hard filtering is not accepted by
the normalized configuration in SDV-07.

### Suggested primary files

- `src/smftools/config/default.yaml`
- `src/smftools/config/experiment_config.py`
- `src/smftools/preprocessing/partitioned_executor.py`
- `src/smftools/cli/preprocess_adata.py`
- `src/smftools/cli/recipes.py`
- `src/smftools/cli/hmm_adata.py`
- partitioned preprocess/full/HMM overlay tests

### Required tests

- Variant evidence is calculated for reads that later fail read,
  modification, or dedup masks.
- Report mode does not change the pre-existing `passes_qc` or `passes_dedup`
  membership.
- `passes_variant_qc` and `passes_nonvariant_qc` exist with documented types.
- Full workflow downstream stages see enabled variant annotations without a
  standalone variant run.
- Disabled mode does not make unused reference/threshold settings mandatory.
- A reporting-only upgrade can reuse compatible prior preprocess work.

### Exit gate

Enabled variant reporting is an ordinary early preprocess analysis available to
the full partitioned workflow, while default filtering behavior remains stable.

## SDV-08 — variant cohort metrics and plots

### Scope

- Define durable named cohorts:
  - `all_aligned`;
  - `pre_dedup_nonvariant_qc`;
  - `post_dedup_nonvariant_qc`;
  - `pre_dedup_final_qc`; and
  - `post_dedup_final_qc`.
- Publish a long-form versioned `variant_qc_metrics.parquet` plus compact
  JSON/TSV summaries.
- Include explicit numerator, denominator, value, cohort, grouping fields,
  reference-set ID, analysis version, and source generation.
- Report read- and duplicate-cluster-level event measures.
- Keep transition/breakpoint events separate from broad other-reference
  evidence.
- Report mixed-status duplicate clusters and event-positive cluster retention.
- Generate plots from the durable metrics/evidence artifacts.
- Register plots through the preprocess plot catalog.
- Bound materialization and plotting using PL-18 resource conventions.
- Provide compact output suitable for a future MultiQC parser.

### Required metric tests

- Numerators and denominators match hand-calculated fixtures for every cohort.
- Variant filtering cannot make its own pre-filter numerator disappear.
- Read-level and cluster-level rates differ in expected duplicate fixtures.
- Mixed-status clusters and keeper outcomes are counted once.
- No-call reads are included/excluded from callable denominators as documented.
- Plot regeneration does not rerun compatible evidence computation.
- Empty cohorts produce defined NaN/zero behavior rather than division errors.

### Exit gate

Pre/post-dedup variant-resolved chimera metrics are durable, reproducible data
products; plots are consumers rather than the sole representation.

## SDV-09 — evidence-gated variant QC and duplicate policy

### Scope

- Introduce filter-mode validation and evidence thresholds.
- Define a strict QC classification separate from existing broad diagnostic
  columns.
- Support minimum informative observations/callable fraction and explicitly
  allowed/disallowed event classes.
- Add stable variant-QC reason codes.
- Define whether duplicate clustering considers:
  - only `passes_qc` reads; or
  - `passes_nonvariant_qc` reads with keeper preference for variant-pass
    members.
- Preserve cluster/event accounting even when members fail final variant QC.
- Recompute combined QC, duplicate clusters, cohort metrics, and affected plots
  when the variant policy changes.
- Propagate changed preprocess channel fingerprints to downstream stages.
- Reconcile deaminase chimera label-only versus exclusion semantics without
  silently coupling the two biological classifiers.

### Required scientific decisions before implementation

- Which variant evidence classes, if any, are excluded in filter mode?
- What minimum informative-site/callability evidence is required?
- Does a fully discordant read without a breakpoint fail, remain diagnostic, or
  trigger an ambiguity/reference-assignment status?
- Are per-read indels part of the initial QC contract?
- Which duplicate candidate/keeper policy is desired?
- Does deaminase PCR-chimera labeling remain independent and label-only?

### Required tests

- One isolated informative call cannot become a large hard-filter event without
  satisfying evidence thresholds.
- Every threshold boundary has pass/fail/no-call coverage.
- Report and filter modes produce identical evidence but different masks only
  where expected.
- Variant failure reasons survive final exclusion.
- Keeper choice is deterministic and follows the chosen policy.
- A variant policy change invalidates combined QC, dedup, and downstream
  consumed channels.
- Spatial/HMM/latent rerun from the new preprocess generation when membership
  changes.

### Exit gate

Variant filtering is explicit, evidence-gated, reproducible, and connected to
dedup/downstream invalidation; it is never inferred from the presence of a
legacy chimera flag.

## SDV-10 — project semantic planning and invalidation

### Scope

- Define project graph adapters using the shared semantic-node model.
- Classify project commands as registry mutations, analysis nodes, or read-only
  consumers.
- Publish deterministic project source snapshots from active registry entries
  and selected experiment stage/generation/channel identities.
- Add a read-only project plan command or equivalent machine-readable option.
- Represent genomic materialization/sample analysis/project embedding
  dependencies explicitly.
- Preserve PL-20 local-latent scope and PL-21 embedding generation behavior.
- Make project products invalidate when a consumed experiment mask/feature
  channel changes.
- Allow reporting-only variant additions to remain compatible for project
  consumers that do not declare those channels.
- Treat existing-molecule membership/feature changes as PL-21 full-refit cases,
  not compatible growth.
- Ensure project registry refresh occurs only after complete experiment
  generation publication.

### Suggested primary files

- new `src/smftools/pipeline/project_graph.py`
- `src/smftools/cli/project_cmd.py`
- `src/smftools/cli_entry.py`
- `src/smftools/project/registry.py`
- `src/smftools/project/catalog.py`
- `src/smftools/project/sample_analysis.py`
- `src/smftools/project/embedding_store.py`
- focused project planner/invalidation tests

Do not rewrite project embedding algorithms or pool experiment-local latent
coordinates.

### Required tests

- Source snapshots are invariant to registry serialization order.
- Duplicate bare read IDs remain collision-free.
- A reporting-only variant channel does not invalidate a consumer that does not
  use it.
- A changed `passes_dedup`/feature channel invalidates affected
  materialization/sample-analysis/embedding results.
- Existing-molecule source changes require the PL-21 refit path.
- A pure new-molecule addition retains existing PL-21 compatible-growth
  behavior when all prior source fingerprints match.
- Project planning is read-only and explains each invalidation.
- Relocated projects and experiments preserve snapshot resolution.

### Exit gate

Experiment and project analysis planning use one compatibility vocabulary, and
project reuse cannot accept stale experiment membership/features.

## SDV-11 — standalone variant command migration

### Scope

- Change `smftools experiment variant` into a compatibility target that requests
  integrated preprocess variant evidence, metrics, and plots.
- Emit a clear deprecation/migration notice for the standalone variant H5AD
  stage.
- Preserve a documented legacy reader for existing variant H5ADs where useful.
- Stop treating legacy `*_performed` booleans as proof of new-node
  compatibility.
- Define behavior for:
  - partitioned raw/preprocess inputs;
  - legacy raw H5AD;
  - legacy non-deduplicated preprocess H5AD;
  - legacy deduplicated H5AD; and
  - missing original raw data.
- Remove the legacy partitioned-stage ambiguity only after project and stage
  resolution tests cover the new authoritative source.
- Update `batch variant` to request the same integrated target during the
  compatibility period.

### Required tests

- The compatibility command produces the same integrated artifacts as
  preprocess/full.
- Repeated invocation plans compatible nodes rather than relying on output path
  existence.
- A legacy deduplicated object reports retained-row limitations.
- Missing filtered-out rows are never represented as a complete all-molecule
  upgrade.
- Project materialization resolves variant annotations from preprocess.
- Deprecation output identifies the replacement command/config behavior.

### Exit gate

There is one authoritative variant implementation and artifact lineage. The old
command remains a safe compatibility interface rather than a second stage.

## SDV-12 — workflow-facing CLI and artifact contract

### Scope

- Add a stable result JSON contract for experiment/project analysis commands.
- Include target, plan/result outcome, run/output root, generation/result IDs,
  artifacts, schemas, checksums, timings, and structured failure information.
- Add a read-only validation command that exits nonzero for incomplete,
  incompatible, stale, or corrupt generations.
- Add explicit task-local output-root overrides and guarantee write ownership.
- Accept staged input paths without rewriting the source config in place.
- Add stable software/tool/model version output.
- Add resource overrides that can map external `task.cpus`, memory, and
  accelerator decisions while retaining the current resource envelope as a
  ceiling.
- Define strict workflow mode for requested optional tools/reports.
- Keep all internal artifact pointers relative and relocation-safe.
- Document which source paths/URI schemes are supported.

### Required tests

- A command writes only inside its declared output root.
- Staged inputs are never modified.
- Success, compatible skip, and failure produce valid result JSON.
- Validation catches missing/corrupt artifacts and pointer mismatches.
- Version output includes smftools and every actually invoked external tool.
- Resource overrides are bounded and recorded.
- Relocated published outputs validate.
- Concurrent experiment runs do not share a current pointer or writable root.

### Exit gate

An external engine can invoke and validate one smftools experiment without
parsing logs, inspecting internal Python objects, or allowing input mutation.

## SDV-13 — production CPU container

### Scope

- Add a production Dockerfile separate from the development container.
- Build from a released/pinned smftools wheel or exact source revision.
- Include the dependencies required for the initial aligned-BAM experiment
  profile.
- Include `/bin/bash` and `ps` for Nextflow runtime compatibility.
- Run as a non-root user and write only to mounted task/output locations.
- Use immutable version tags/digests; never rely on `latest`.
- Emit image labels and an SBOM and run dependency/image scanning in CI.
- Add a tiny container smoke fixture covering version, plan, full execution,
  result JSON, and validation.
- Document external-tool/model licensing and whether tools are bundled or
  supplied through separate workflow processes.
- Keep Dorado/CUDA/GPU support out of the CPU image unless separately justified.

### Required tests

- `smftools --version` works in the built image.
- The BAM-entry tiny experiment runs without undeclared host executables.
- Outputs are writable under Docker and Apptainer-compatible user mappings.
- The result/version artifacts identify the image/tag/digest when supplied.
- Re-running the fixture against the same read-only inputs is deterministic at
  the declared scientific-result level.

### Exit gate

A pinned CPU image can run the supported BAM-entry full workflow in an isolated
task directory and produce independently validated outputs.

## SDV-14 — documentation and integrated acceptance

### Documentation

Update CLI, configuration, storage, migration, project, and workflow
documentation to cover:

- semantic versus execution DAG responsibilities;
- experiment/project plan output;
- analysis compatibility states and force behavior;
- immutable preprocess generations and rollback;
- variant raw/preprocess ownership;
- reference-set and variant-call semantics;
- reporting versus filtering modes;
- QC masks/reason fields;
- named pre/post-dedup metric cohorts;
- standalone variant command migration;
- legacy upgrade limitations;
- downstream/project invalidation;
- workflow result/validation/version contracts; and
- production container usage.

Before editing documentation, re-read `docs/source/AGENTS.md` and run the
warnings-as-errors Sphinx build.

### Integrated acceptance matrix

| Dimension | Values |
|---|---|
| Experiment count | one; two with duplicate bare read IDs |
| Analysis mode | locus; genome |
| Variant mode | off; report; filter |
| Reference sets | valid pair; multiple sets; missing/ambiguous invalid set |
| Variant evidence | no-call; self; fully discordant; one breakpoint; multiple breakpoints; tile-boundary breakpoint |
| QC cohort | all aligned; non-variant pre/post dedup; final pre/post dedup |
| Duplicate cluster | none; homogeneous event; mixed event; keeper preference |
| Lifecycle | fresh; compatible restart; missing new node; config change; algorithm change; source change; corruption; injected failure |
| Upgrade source | partitioned raw; prior preprocess generation; legacy raw H5AD; legacy pp; legacy dedup |
| Downstream | spatial; HMM; latent; project materialization; project embedding |
| Filesystem | original; relocated experiment; relocated project |
| External execution | host; Docker; Apptainer-compatible profile |

### Required command families

Use `venvs/venv-all` unless the active environment already satisfies the task:

```text
venvs/venv-all/bin/python -m pytest -q <focused semantic/variant/preprocess/project tests>
venvs/venv-all/bin/python -m pytest -m unit -q
venvs/venv-all/bin/python -m pytest -m integration -q
venvs/venv-all/bin/python -m pytest -m smoke -q
venvs/venv-all/bin/ruff check .
venvs/venv-all/bin/ruff format --check .
sphinx-build -W -b html docs/source docs/_build/html
```

Run relevant E2E, failure-injection, relocation, and container tests for each
touched public path. Record optional-dependency/platform exclusions rather than
weakening assertions.

### Exit gate

All program findings have automated acceptance or an explicitly approved
deferment; CLI help and documentation describe the actual graph, variant,
upgrade, and container behavior.

## External workflow lane

External workflow code should live in a separate repository. These are
coordination milestones rather than smftools core PRs.

### NF-01 — independent Nextflow DSL2 prototype

#### Scope

- Start from aligned, modification-bearing BAM input.
- Use one local `SMFTOOLS_FULL` process per experiment.
- Render the smftools config inside the Nextflow task directory.
- Set a fresh task-local output root.
- Pin the SDV-13 production image by version/digest.
- Emit the complete experiment bundle, result JSON, manifest, versions, and
  compact QC outputs.
- Use an nf-core-style samplesheet and parameter JSON Schema even before
  requesting nf-core adoption.
- Add tiny `test` and representative `test_full` profiles.
- Test `-resume`, failure propagation, relocation, Docker, and Apptainer.
- Do not invoke `smftools experiment batch`.
- Do not split raw/preprocess/spatial/HMM/latent into separate processes while
  those commands require a shared mutable run tree.

#### Exit gate

Nextflow can scatter independent experiments, resume unchanged work, and
publish validated relocation-safe bundles without modifying staged inputs.

### NF-02 — nf-core readiness and possible proposal

#### Scope

- Discuss scope and ownership with the nf-core community before transfer.
- Generate/align the repository with the current nf-core template.
- Add nf-test coverage, lint, documentation, release, and MultiQC integration.
- Emit versions for smftools and every invoked external tool.
- Decide whether smftools should first be packaged through Bioconda,
  BioContainers, or a maintained custom image.
- Keep multi-tool orchestration as a local module/subworkflow until its
  input/output contract is stable.
- Contribute a reusable `nf-core/modules` component only after the smftools
  module has stable inputs, outputs, versions, tests, and stubs.
- Add separate basecalling/GPU processes and images only after the BAM-entry
  pipeline is stable.

#### Exit gate

The pipeline satisfies current nf-core requirements and has community agreement
on whether it should enter nf-core governance.

## Schema and migration policy

- Version the semantic plan, node result, preprocess generation, reference-set,
  variant task catalog, variant read/event index, variant sidecar, QC metric,
  result JSON, and versions schemas independently.
- Readers support older schemas only when identity and semantics can be
  recovered without guessing.
- A bare legacy `*_performed=True` flag is never a semantic cache hit.
- A legacy deduplicated H5AD cannot claim all-molecule coverage.
- Raw/reference source changes invalidate dependent evidence.
- Reporting-only additions may reuse compute but still publish a new immutable
  generation.
- QC/dedup changes invalidate declared downstream consumers.
- Old complete generations remain available until an explicit future retention
  policy removes them.
- User-facing config, CLI, mask, or default changes require migration notes.

## Explicit non-goals

- Embedding Nextflow or Snakemake in smftools.
- Building a new cluster/cloud scheduler.
- Making physical task chunking part of scientific identity.
- Exposing private Python workers directly to an external workflow.
- Editing a published experiment or preprocess generation in place.
- Treating file/column existence alone as compatibility.
- Retrofitting variant evidence to change raw alignment assignment.
- Making `chimeric_variant_sites` a default filter without evidence validation.
- Pooling independently fitted experiment-local latent coordinates.
- Rewriting PL-15 through PL-23 implementations without a demonstrated defect.
- Adding a complete multi-allelic probabilistic caller unless separately
  designed.
- Bundling Dorado/CUDA/vendor model assets into the initial CPU image.
- Publishing an nf-core pipeline from this repository.
- Bumping the package release version on feature branches.

## Decision gates

The following choices do not block SDV-01 through SDV-04. They must be resolved
before the named PR:

| Decision | Needed by |
|---|---|
| Initial variant model: one pair with extensible schema versus multiple allele/reference members immediately | SDV-05 |
| Per-read indels supported versus explicitly excluded from the first calling contract | SDV-05 |
| Variant integration default: explicit `off` versus automatic `report` when references are configured | SDV-07 |
| Exact hard-filter evidence thresholds and allowed event classes | SDV-09 |
| Broad discordance without a breakpoint: diagnostic, QC failure, or reference-assignment ambiguity | SDV-09 |
| Duplicate candidate and keeper policy in the presence of variant failures | SDV-09 |
| Deaminase PCR-chimera label-only versus exclusion behavior | SDV-09 |
| Initial external workflow owns BAM-entry only versus raw signal/basecalling | NF-01 |
| Independent Nextflow maturity threshold before proposing nf-core governance | NF-02 |

## Program completion definition

The program is complete when:

1. Experiment and project analysis commands use one semantic compatibility
   vocabulary and explainable planner.
2. Existing direct-stage commands remain valid target aliases.
3. Preprocess generations are immutable, transactional, validated, and
   restart-safe.
4. A prior partitioned experiment can receive a new compatible analysis without
   mutating old outputs or rerunning unrelated nodes.
5. Variant reference/evidence semantics are explicit, versioned, and
   reference-set aware.
6. Variant evidence covers all available raw molecules before filtering.
7. Reporting mode preserves prior QC/dedup membership.
8. Filter mode is evidence-gated and publishes independent masks/reasons.
9. Variant-resolved event metrics have explicit pre/post-dedup denominators at
   read and cluster levels.
10. QC/dedup changes invalidate affected spatial, HMM, latent, and project
    products.
11. The standalone variant command is a safe compatibility interface to the
    integrated implementation.
12. External callers receive stable result, validation, versions, resource,
    and output-ownership contracts.
13. A pinned CPU container passes the BAM-entry smoke workflow.
14. Focused, unit, integration, smoke, relocation, failure-injection, lint,
    format, documentation, and applicable container gates pass.

## Implementation status

| PR | Status | Branch | Notes |
|---|---|---|---|
| SDV-01 | Merged | `feature/semantic-analysis-graph` | PR #417; merge `d88adf8` |
| SDV-02 | Merged | `feature/experiment-semantic-planning` | PR #418; merge `b11cf74` |
| SDV-03 | Merged | `feature/preprocess-immutable-generations` | PR #419; merge `3828e32` |
| SDV-04 | Merged | `feature/preprocess-incremental-upgrades` | PR #420; merge `c0fc57f` |
| SDV-05 | Merged | `feature/variant-reference-contract` | PR #421; merge `e0d0d69` |
| SDV-06 | Merged | `feature/partitioned-variant-evidence` | PR #422; merge `7a47911` |
| SDV-07 | Merged | `feature/preprocess-variant-reporting` | PR #423; merge `992ccb0` |
| SDV-08 | Merged | `feature/variant-cohort-metrics` | PR #424; merge `9a0f088` |
| SDV-09 | Merged | `feature/variant-qc-dedup-policy` | PR #425; merge `d906e28` |
| SDV-10 | Merged | `feature/project-semantic-planning` | PR #426; merge `afc8af3` |
| SDV-11 | Merged | `fix/variant-command-preprocess-alias` | PR #427; merge `abbe4fd` |
| SDV-12 | Merged | `feature/workflow-cli-contract` | PR #428; merge `5e019e8` |
| SDV-13 | Merged | `feature/production-cpu-container` | PR #429; merge `fd69d7a` |
| SDV-14 | Merged | `feature/semantic-variant-acceptance` | PR #430; merge `4e1b1e5` |
| NF-01 | Future external repository | — | Begin after SDV-12/SDV-13 and stable reporting outputs |
| NF-02 | Future external repository | — | Requires NF-01 maturity and nf-core discussion |

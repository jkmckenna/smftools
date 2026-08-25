# Variant-to-preprocessing and incremental reprocessing audit

**Audit date:** 2026-07-27
**Repository state reviewed:** `293ec85` on
`feature/project-latent-acceptance`
**Companion plan reviewed:**
`dev/project_and_latent_partitioned_pipeline_implementation_plan.md`, including
completed PL-15 through PL-23
**Scope:** the `smftools experiment variant` and `preprocess` commands, raw
alignment and sequence extraction, legacy and partitioned preprocessing,
variant-derived QC and duplicate metrics, stage metadata/provenance, and the
feasibility of applying newly added preprocessing analyses to prior outputs.
The audit also evaluates internal analysis DAGs, Snakemake/Nextflow
compatibility, production containers, and a possible future nf-core pipeline.
**Out of scope:** implementation. This document is an audit and design
recommendation only.

## Executive assessment

Absorbing the current variant functionality into preprocessing is both feasible
and architecturally preferable, especially for the partitioned pipeline. The
standalone variant stage is a legacy-only side branch: it requires a monolithic
preprocessed H5AD, runs after QC and duplicate removal, is absent from the
standard `full` workflow, has no partitioned executor or lifecycle record, and
cannot influence preprocessing QC. Meanwhile, most of the data it needs already
exists at the raw/partitioned boundary.

The migration must preserve an important boundary:

- read alignment to the multi-record FASTA and secondary-alignment rescue happen
  before raw extraction and should remain there;
- alignment of reference sequences to each other, informative-site discovery,
  per-read variant calls, and per-read variant/chimera annotations can move into
  preprocessing before read QC; and
- population summaries and plots should be generated from explicitly named
  cohorts after the relevant QC and duplicate masks are available.

The current variant implementation is not yet safe to use as a hard QC filter
without refinement. In particular, `chimeric_variant_sites` means that a read has
*any* contiguous segment assigned to the other reference, whereas the
pair-specific `*_is_chimeric` flag means that the variant class changes within
the read. A single informative discordant call can be extended across a large
part of the read span by the segmentation algorithm. These labels are valuable
diagnostics, but hard-filter defaults should be introduced only with minimum
callability/evidence thresholds and validated data.

Before/after-duplicate metrics are straightforward in the partitioned model
because it retains every raw molecule and stores `passes_read_qc`, `passes_qc`,
`is_duplicate`, and `passes_dedup` as masks. The same row-level variant
annotations can therefore be summarized over multiple cohorts without deleting
evidence. The metrics should distinguish read counts, event rates, unique
duplicate clusters, event types, and within-cluster discordance.

Applying updated preprocessing code to a prior output is also possible, but the
package does not currently provide the required analysis-level compatibility
contract:

- legacy H5AD files have useful history and `*_performed` flags, but those flags
  do not identify parameter values, algorithm versions, inputs, or output
  checksums;
- partitioned stages have config hashes, artifacts, task catalogs, schema
  versions, and lifecycle records, but compatibility is decided for the whole
  preprocess stage rather than for individual analyses;
- software/algorithm version is not part of preprocess skip compatibility;
- recorded schema versions are not matched by the preprocess skip check; and
- the partitioned preprocess executor rewrites one canonical generation instead
  of appending a new analysis generation or reusing compatible task outputs.

The recommended direction is an analysis-DAG model on top of the existing
partitioned raw store and *inside the shared stage lifecycle established by
PL-17*. Each analysis needs a stable ID, explicit algorithm/schema version,
semantic config subset, input artifact identities, dependencies, task scope,
and output contract. A planner can then classify work as compatible, missing,
stale, or blocked. A missing independent analysis may be added by publishing a
new immutable preprocess generation that reuses compatible artifacts; a
published generation must never be edited in place. Analyses that alter
QC/dedup masks must create a new preprocess generation and invalidate dependent
spatial/HMM/latent generations.

PL-15 through PL-23 are now implemented at the repository state reviewed. This
materially strengthens the proposal: collision-free molecule identity,
transactional latent/project generations, resource plans, generation-scoped
indexes, scoped project latent access, source-sensitive embeddings,
deterministic model provenance, and full-workflow acceptance are available as
working patterns rather than future assumptions.

For workflow orchestration, the recommendation is **not** to embed Snakemake or
Nextflow inside the Python package and **not** to build a competing distributed
scheduler. smftools should implement a small internal dependency graph for
scientific analysis semantics and incremental compatibility. An external engine
should own execution DAG concerns such as staging, containers, retries,
cluster/cloud submission, and parallel experiments. Because nf-core is an
eventual goal, Nextflow should be the first reference integration; the
underlying CLI/artifact contract should remain equally usable from Snakemake.

## Compatibility with the project/latent partitioned-pipeline plan

### Compatibility result

The recommendations in this audit are compatible with
`project_and_latent_partitioned_pipeline_implementation_plan.md` after one
important refinement: preprocessing should extend the stage lifecycle and
generation conventions already implemented for latent in PL-17, not introduce
a parallel publication/cache system.

| Companion-plan contract | Required preprocessing/variant interpretation |
|---|---|
| PL-15 project-global molecule identity | Persist and join variant outputs by `molecule_uid`, with `experiment_uid` and original `read_id` retained. Bare `read_id` is allowed only inside a verified single-experiment scope. |
| PL-16 fail-fast configuration/source contract | Validate enabled variant reference sets, calling policies, and QC thresholds in `ExperimentConfig`. Disabled filters should not make unused settings mandatory. The compatibility alias for `variant` requests preprocess analyses; it does not create another partitioned source-stage type. |
| PL-17 manifest-driven transactional lifecycle | Build under `.staging/<generation_id>`, validate all artifacts, atomically publish `generations/<generation_id>`, update the canonical spine and `current.json`, then mark the experiment-manifest stage complete. A failed update leaves the prior generation current. |
| PL-17 compute/plot separation | Variant computation, QC masks, and cohort metrics are compute-affecting. Plot configuration is separate and may publish a new generation that reuses validated compute artifacts. |
| PL-18 resource planning | Variant reference planning, molecule calls, reducers, writes, and plots need pre-allocation estimates. Machine-specific worker/chunk choices are execution provenance and must not redefine portable logical task identity. |
| PL-19 molecule index | Reuse the collision-free identity and portable Parquet-index pattern. Variant records require their own scope key; they must not be inserted into the latent index or inherit latent coordinate semantics. |
| PL-20 scoped latent access | Variant calls and QC masks are genomic/per-molecule data and may participate in ordinary genomic materialization. They must not cause experiment-local latent coordinates to be attached or pooled. |
| PL-21 project embedding provenance | A changed variant QC/dedup mask changes project membership and therefore makes an existing embedding source incompatible. It is not “pure growth” and requires the refit behavior specified by PL-21. |
| PL-22 immutable model/provenance IDs | Apply the same explicit implementation/schema-version principle to variant analyses and reference catalogs, while keeping variant evidence IDs distinct from latent model IDs. |
| PL-23 acceptance and migration | Add variant/preprocess cases to lifecycle, relocation, identity-collision, source-change, failure-injection, and project-selection coverage rather than creating a separate acceptance track. |

All of PL-15 through PL-23 are implemented at the repository state reviewed:

- PL-15 provides project-safe `experiment_uid`/`molecule_uid` joins.
- PL-16 provides fail-fast latent configuration and deterministic source
  selection.
- PL-17 provides immutable transactional latent generations and compute/plot
  reuse.
- PL-18 publishes versioned resource estimates and effective runtime decisions.
- PL-19 publishes a generation-scoped latent molecule index (schema 3 at
  `293ec85`).
- PL-20 provides indexed, scoped latent access/export without pooling local
  coordinate systems.
- PL-21 publishes immutable project-embedding generations with source and
  feature fingerprints.
- PL-22 provides deterministic fit membership, immutable model IDs/bundles, and
  compatible-growth reuse.
- PL-23 adds integrated acceptance/migration guidance and runs latent after HMM
  in `experiment full` by default, with an explicit opt-out.

The variant/preprocess work should reuse these implemented interfaces and
schema patterns. It no longer needs to preserve speculative extension points
for unfinished PL work.

### One lifecycle, with analysis-level reuse

The shared contract should have two nested levels:

1. The **experiment manifest** remains authoritative for whether the preprocess
   stage is complete and which immutable preprocess generation is current.
2. The **preprocess generation manifest** enumerates the analysis nodes and
   task artifacts that compose that generation, including their compatibility
   keys and reused source generation IDs.

“Append” must not mean adding files beneath the directory named by the current
generation. It means:

1. resolve and validate the current preprocess generation;
2. classify its analysis nodes as reusable, missing, or stale;
3. create a unique staging generation;
4. copy or hard-link validated compatible artifacts into the staged generation,
   or reference an immutable content-addressed artifact only when its retention
   dependency is explicit and relocation-safe;
5. compute only missing/stale nodes and their dependents;
6. validate the complete staged generation, including indexes and relative
   pointers;
7. atomically publish it and advance `current.json`; and
8. regenerate the consolidated experiment spine only after publication.

This is the preprocessing equivalent of PL-17 compute reuse for a plot-only
latent change. It preserves the old complete generation for rollback and avoids
two independent notions of “current.”

### Identity and ownership

Variant task/index identity should follow the companion plan's collision-free
rules:

```text
per-molecule variant result:
  (experiment_uid, molecule_uid, variant_reference_set_id,
   analysis_generation_id)

position overlay:
  (experiment_uid, molecule_uid, variant_reference_set_id,
   analysis_core_id, analysis_generation_id)
```

This differs intentionally from PL-19 latent uniqueness, where one molecule may
belong to several independently fitted coordinate owners. Variant per-molecule
classification must have one owner across core boundaries; only its positional
overlay may be core-partitioned. Original `read_id` remains a traceability
field, not a global key.

The logical identity must not include memory-derived worker count, barcode
chunk size, or transform/write chunk size. Those belong in execution/resource
provenance, consistent with PL-18. A reference-set ID should use canonical
reference identities, orientation, sequence checksums, alignment/scoring
semantics, and informative-site policy so it is stable across relocated
experiments without relying on a display name alone.

### Downstream invalidation and project behavior

The companion plan makes source generations part of latent and project
compatibility. The resulting rules are:

- Reporting-only variant evidence can reuse all prior preprocess compute, but
  publishing it still creates a new preprocess generation.
- Changing `passes_variant_qc`, `passes_qc`, duplicate candidates, or duplicate
  keepers changes the scientific source consumed by downstream stages. Spatial,
  HMM, and latent outputs derived from the old preprocess generation are stale.
- If latent consumes spatial or HMM rather than preprocess directly, that
  intermediate stage must first be regenerated; the new upstream generation
  then flows into latent's PL-17 source fingerprint.
- A project embedding whose membership or features change for existing
  molecules follows PL-21's explicit full-refit rule. A diagnostic addition is
  not a “new molecule” extension.
- The project registry/consolidated spine should be refreshed only after the new
  preprocess generation is complete, preserving the PL-17/PL-M4 publication
  order.

At first, the existing whole-source fingerprint may conservatively invalidate a
downstream stage even when the appended variant output is diagnostic-only and
none of that stage's consumed masks/layers changed. That is safe and compatible
with PL-17. Avoiding the extra recomputation later would require
dependency-specific input fingerprints declared by both producer and consumer;
it must not be implemented by ignoring a changed upstream generation.

Variant fields exposed through project genomic materialization remain ordinary
read/genomic annotations. Nothing in this proposal changes the companion plan's
central latent rule: independently fitted local latent coordinates are never
pooled, and project-global embeddings remain a separate product.

### Revised implementation sequencing after PL-23

The prior branch-conflict risk is gone. The implementation sequence should now
be:

1. Generalize the completed PL-17 publication pattern to preprocess:
   `StageLifecycle`, immutable staging/generation directories, validation,
   canonical spine, `current.json`, and consolidated-spine publication.
2. Land variant reference/evidence semantics and reporting-only QC before
   enabling hard filtering.
3. Add a versioned variant/preprocess index using PL-19's collision-free,
   bucketed, relocation-safe Parquet conventions without changing latent-index
   row meaning.
4. Reuse PL-18 resource-envelope and estimator records for variant molecule
   tasks, reducers, serialization, and plotting.
5. Enable variant QC/dedup effects with source-generation invalidation tests for
   spatial, HMM, latent, and PL-21 project embeddings.
6. Add algorithm/reference-set compatibility IDs in PL-22's explicit
   implementation/schema-version style, while keeping variant evidence IDs
   separate from latent model IDs.
7. Extend PL-23's acceptance map with variant, preprocess-upgrade, external
   workflow, container, and relocation cases.

This builds on the completed lifecycle, identity, resource, index, and
provenance contracts instead of reopening them.

## Workflow orchestration, DAG, and nf-core assessment

### Recommendation

Use three deliberately separate layers:

```text
Nextflow/nf-core or Snakemake execution DAG
  - samples, files, containers, CPU/GPU/memory, retries, HPC/cloud, publishing
                     |
                     v
stable engine-neutral smftools CLI and artifact contract
                     |
                     v
smftools semantic analysis DAG
  - algorithm/schema versions, scientific dependencies, compatibility,
    QC/dedup invalidation, generation validation
                     |
                     v
bounded partition planners/executors and reducers
```

The internal DAG should be a dependency and compatibility model, not a general
workflow engine. It answers:

- which scientific analyses depend on which inputs/masks;
- which implementation/config/input identity produced an artifact;
- whether an old node is compatible, missing, stale, or corrupt;
- what must be recomputed when variant QC changes; and
- what belongs in one validated preprocess generation.

Nextflow or Snakemake answers:

- where the command runs;
- which container and executor it uses;
- what files are staged;
- how many CPUs, memory, GPUs, and time it receives;
- how experiments are scattered;
- whether a failed task is retried; and
- which declared outputs are published.

Reimplementing those execution concerns inside smftools would duplicate mature
scheduler, cache, container, and cloud behavior. Conversely, asking Nextflow or
Snakemake to infer scientific compatibility from the presence of an `obs`
column would discard the package's stronger domain provenance. The two DAGs are
complementary.

### Nextflow versus Snakemake

Both engines can wrap a well-behaved CLI. The package should not import either
engine or emit engine-specific artifacts from its scientific core.

| Consideration | Nextflow | Snakemake | Implication for smftools |
|---|---|---|---|
| Primary abstraction | Dataflow channels and isolated processes | File/rule targets and wildcards | Expose explicit files/directories, values, and stable outputs. |
| Execution environments | Local, HPC, cloud, and multiple container runtimes | Local/HPC/cloud via executor and storage plugins; Conda and Apptainer deployment | Keep compute logic outside engine files and containers self-contained. |
| Reuse unit | DSL2 module/subworkflow | Wrapper, module, or included workflow | One stable CLI can support both wrappers. |
| Restart | Task hash plus work-directory outputs with `-resume` | Output/rule provenance and rerun triggers | Do not make engine inputs mutable; retain smftools manifests for scientific validation. |
| Community target | nf-core is built on Nextflow | Snakemake Wrapper/Workflow catalogs | Prioritize a Nextflow reference pipeline because nf-core is the stated destination. |

Nextflow stages declared `path` inputs into a unique task work directory and
expects declared outputs to be produced there. Its documentation explicitly
calls modifying process inputs an anti-pattern because it breaks resume
behavior ([Nextflow processes](https://www.nextflow.io/docs/latest/process.html),
[caching and resuming](https://www.nextflow.io/docs/latest/cache-and-resume.html)).
Snakemake can provide an equivalent wrapper using rules, modules, and
per-rule Conda/Apptainer environments
([Snakemake modularization](https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html),
[deployment](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html)).

Nextflow is therefore the recommended first orchestration implementation, but
the prerequisite work is engine-neutral CLI/output hardening rather than
Nextflow code in this repository.

### Current package readiness

At `293ec85` / `2.17.0.dev0`, smftools is **coarse-process ready with an
adapter**, but **not yet nf-core-ready** and not yet suitable for one external
process per internal partition.

| Contract | Current state | Assessment |
|---|---|---|
| Versioned command-line tool | `smftools --version` emits `2.17.0.dev0`; Click exposes experiment and project commands | Strong |
| One-command experiment | `smftools experiment full CONFIG` runs raw -> preprocess -> spatial -> HMM -> latent | Strong for one process per experiment |
| Stage entry points | Raw, preprocess, spatial, HMM, and latent can run independently | Useful, but all write sibling paths under one configured run root |
| Failure signaling | Stage exceptions propagate; batch exits nonzero when any experiment fails | Strong |
| Machine-readable completion | `full_summary.json`, `experiment_manifest.json`, generation manifests, task catalogs, and JSONL performance logs exist | Strong data, but no uniform `--result-json` contract on every command |
| Scientific restart/provenance | Stage manifests, checksums, config/source hashes, PL-17/PL-21 immutable generations, PL-22 model IDs | Strong |
| Collision-free identity | PL-15/PL-19 persist experiment and molecule identity | Strong |
| Resource awareness | CPU affinity, cgroup and scheduler limits are detected; PL-18 records effective latent decisions | Strong inside an allocated process |
| Portable storage | Relative run-root pointers, Parquet indexes, Zarr task stores, and relocation tests | Strong on filesystems; direct object-store URIs are not a current stage-write contract |
| Isolated output ownership | `experiment full` can write a fresh run tree; individual stages mutate/add siblings in the same tree | Partial |
| Read-only process inputs | Raw instrument data is treated as read-only, but a stage-by-stage external workflow would need to carry forward and mutate/copy a run tree | Weak for fine-grained external processes |
| Production container | Only `.devcontainer/Dockerfile` exists; it is a development image and does not install the completed package/toolchain as a release artifact | Missing |
| Reproducible software lock | Python dependencies have bounded ranges and releases are on PyPI, but there is no production image digest or workflow lock | Partial |
| External-tool provenance | Dorado version is inspected in selected paths; there is no unified versions artifact for Dorado/minimap2/modkit/samtools/bedtools/MultiQC | Partial |
| Workflow input schema | Experiment configuration is a variable/value CSV with many path-bearing fields; batch consumes paths to those configs | Functional locally, not an nf-core-style samplesheet/JSON Schema |
| Workflow-level tests | Python unit/integration/smoke, relocation, resource, and failure-injection coverage are strong | No container smoke test, Nextflow test, nf-test, or nf-core lint yet |
| MultiQC | An optional best-effort BAM-QC invocation exists | No smftools module producing nf-core-ready custom-content summaries across variant/preprocess metrics |

The completed PL work is unusually helpful for an external workflow: outputs
are self-describing, identity-safe, resource-aware, and validated. The remaining
work is mostly packaging and process-boundary design, not a rewrite of the
analysis algorithms.

### Recommended process granularity

#### Near term: one Nextflow process per experiment

The safest first wrapper is:

```text
tuple(meta, raw input path/directory, reference FASTA, optional sample sheet,
      smftools parameter overrides)
  -> SMFTOOLS_FULL
  -> tuple(meta, complete experiment directory, full_summary.json,
           experiment_manifest.json, versions)
```

The process should render an experiment config inside its task directory, set
`output_directory` to a new task-local directory, run
`smftools experiment full`, validate `full_summary.json` and
`experiment_manifest.json`, and emit the complete experiment directory.
Nextflow then parallelizes experiments through the channel. Do **not** call
`smftools experiment batch` from Nextflow; that would hide per-experiment
resource requests, retries, caching, and failures inside one scheduler task.

This coarse boundary matches Nextflow's isolated work-directory model and
avoids copying a large experiment tree between five process invocations. It
also lets smftools retain its internal bounded task scheduler and stage
lifecycle.

Starting from a basecalled, aligned BAM with modification tags/backends
available is the easiest first container target. POD5/FAST5 basecalling,
demultiplexing, alignment, and modkit extraction introduce vendor tools, GPU
selection, models, and larger containers; they can be added after the BAM-entry
pipeline is stable.

#### Medium term: stage-level processes only with immutable stage I/O

It is tempting to define one external process for raw, preprocess, spatial,
HMM, and latent immediately. The current package makes that inefficient:
downstream stages write sibling artifacts and update the experiment manifest
and consolidated spine under one run root. Passing that run root as a process
input and modifying it would violate Nextflow's resume guidance. Copying the
whole tree into every process would be correct but costly.

Stage-level external processes become attractive after each command can:

- accept upstream generations as read-only declared inputs;
- write a new self-contained output bundle/directory;
- emit a manifest that identifies upstream generations without requiring a
  mutable shared root;
- leave final `current.json`/registry publication to one owning finalizer; and
- avoid absolute links back into a task work directory.

The proposed immutable preprocess generation is a step in this direction, but
raw, spatial, HMM, consolidated-spine, and experiment-manifest ownership must
also have an explicit bundle/finalization contract.

#### Long term: optional partition scatter/gather

Externalizing every reference/core/barcode/read-chunk task could allow
Nextflow to schedule very large experiments across a cluster. It should not be
attempted by calling private Python workers. A supported protocol would need:

```text
smftools experiment plan-stage
  -> versioned plan.json + immutable input artifact identities

smftools experiment run-task
  -> one task-spec input + one task result directory/manifest

smftools experiment finalize-stage
  -> validates all expected results, reduces them, and atomically publishes
     one stage generation
```

Logical task identity must remain independent of executor, worker count, memory
envelope, and physical chunk size. Only the finalizer may advance a stage's
current pointer. This protocol would be usable by Nextflow, Snakemake, Slurm
arrays, or the existing local executor.

Until that protocol exists, smftools should keep partition scheduling internal.
Running both an external scatter and the current internal process pools at the
same granularity would create nested scheduling and oversubscription.

### Engine-neutral CLI and artifact changes

The following changes would make smftools a good workflow tool without coupling
it to an engine:

1. **Stable result JSON for every command.** Add an explicit result path or
   stdout mode containing command/stage, outcome, run root, generation ID,
   required artifacts, checksums, schemas, timings, and failure details.
2. **Explicit output ownership.** Allow an output root override at the CLI and
   guarantee that a command writes only inside it.
3. **Workflow config renderer/validator.** Accept an engine-friendly structured
   parameter object plus staged file paths and render the existing
   variable/value experiment CSV. A JSON Schema should validate the external
   interface separately from the larger internal `ExperimentConfig`.
4. **Read-only upgrade input.** Applying new preprocessing code to an old
   object should accept the old run/generation as an immutable input and publish
   a new run or generation. Never update the staged input directory in place.
5. **Resource overrides.** Provide CLI/environment overrides for threads,
   memory, device/GPU, and plot limits so a wrapper can map `task.cpus`,
   `task.memory`, and `task.accelerator` without rewriting arbitrary config
   fields. Existing cgroup/scheduler detection remains a safety ceiling.
6. **Unified software versions.** Emit smftools, Python, relevant libraries,
   container image/digest when supplied, and every invoked external tool/model
   version in a stable artifact.
7. **Deterministic validation command.** Add a read-only command that validates
   an experiment/stage generation and exits nonzero on an incomplete, stale, or
   corrupt artifact.
8. **URI and relocation policy.** Define whether sources may be filesystem
   paths only or supported object-store URIs; keep internal pointers relative
   and publication-safe.
9. **QC summary contract.** Publish compact JSON/TSV metrics, including the
   proposed variant pre/post-dedup cohorts, for MultiQC and workflow reports
   without reading AnnData/Zarr.
10. **No hidden optional success.** A workflow mode should make requested
    external tools and reports fail deterministically when absent rather than
    silently skipping them. Optional best-effort local behavior can remain a
    separate mode.

These contracts also make a Snakemake wrapper straightforward.

### Interaction between engine caching and smftools compatibility

Nextflow hashes the task name, script, inputs, container/environment, and other
task metadata, and reuses a task only when its cached outputs still exist
([Nextflow caching](https://www.nextflow.io/docs/latest/cache-and-resume.html)).
This is execution reproducibility, not full scientific compatibility.

The intended division is:

- Pinning a new smftools container changes the external task hash and normally
  reruns the process.
- smftools analysis IDs/algorithm versions decide which nodes inside an
  explicitly supplied prior generation may be reused.
- Nextflow never treats a changed staged experiment directory as the same
  input, and smftools never mutates that input to make it look current.
- A fresh nf-core run lets Nextflow resume whole experiment processes.
- An intentional “upgrade old experiment” workflow supplies the old experiment
  as a read-only input and emits a new upgraded experiment/generation.

This avoids ambiguous double caching. The external engine owns whether a
process invocation is reused; smftools owns whether scientific sub-artifacts
are compatible when a prior generation is intentionally provided.

### Production container strategy

The existing devcontainer is not a release container: it preinstalls some heavy
dependencies for editor use, does not install a pinned smftools release as the
final application, and does not establish complete external-tool or image
provenance.

A production plan should provide:

1. **CPU analysis image.** Exact smftools release/wheel plus the Python/runtime
   dependencies required for BAM-entry raw ingestion, preprocess, spatial, HMM,
   latent, variant, plotting, and selected Python I/O backends.
2. **Optional ONT/GPU image or processes.** Dorado/model and CUDA-sensitive
   work should be separate when its image is materially larger. nf-core
   explicitly supports dual CPU/GPU container selection based on
   `task.accelerator` and requires exact CUDA pinning
   ([nf-core GPU/software requirements](https://nf-co.re/docs/specifications/components/modules/software-requirements)).
3. **Pinned immutable references.** Publish versioned tags and digests; never
   `latest`. nf-core requires stable versioned Docker software and recommends
   Bioconda/BioContainers where possible
   ([nf-core Docker requirement](https://nf-co.re/docs/specifications/pipelines/requirements/docker)).
4. **Runtime compatibility.** Include `/bin/bash` and `ps`, which Nextflow
   requires for task execution/metrics
   ([Nextflow containers](https://www.nextflow.io/docs/latest/container.html)).
   Run as a non-root user and write only to mounted task/output paths.
5. **Supply-chain artifacts.** Build from the released wheel, record image
   labels/digest, generate an SBOM, scan dependencies, and smoke-test
   `smftools --version` plus a tiny pipeline fixture.
6. **Tool/model policy.** Verify redistribution/licensing and model-download
   behavior for Dorado and any vendor assets before bundling them. Prefer
   separate existing nf-core modules/containers for external tools when their
   file boundary is stable.

No Bioconda recipe or production smftools image is present in this repository,
and this audit did not find a current smftools Bioconda/BioContainers package.
A Bioconda recipe would be valuable because nf-core can derive container
definitions from it, but the large Python/torch stack and GPU variants may make
a custom pinned image the faster first milestone.

### Practical path to nf-core

An nf-core pipeline should live in a separate repository generated from the
nf-core template, not inside the Python package repository. nf-core pipelines
have their own template synchronization, branch, schema, documentation,
community-ownership, lint, release, and RO-Crate requirements
([pipeline requirements](https://nf-co.re/docs/guidelines/pipelines/overview),
[template structure](https://nf-co.re/docs/developing/pipelines/template-files)).
The smftools Python package remains the versioned scientific tool consumed by
that pipeline.

A sensible progression is:

1. Publish and test a pinned CPU smftools container.
2. Build a non-nf-core-branded Nextflow DSL2 prototype in a separate repository,
   using one `SMFTOOLS_FULL` local module per experiment from an aligned BAM.
3. Define an nf-core-style samplesheet and `nextflow_schema.json`; translate
   rows/parameters into task-local smftools configs.
4. Emit a versions topic entry for smftools and all invoked tools, capture
   `full_summary.json`/manifest outputs, and add compact QC data to MultiQC.
5. Add `-profile test` with a tiny licensed dataset, `test_full`, nf-test
   snapshots, container tests, `-resume` tests, and relocation checks.
   Current nf-core recommendations use nf-test and require a working test
   profile
   ([nf-core testing](https://nf-co.re/docs/specifications/pipelines/recommendations/testing)).
6. Discuss the pipeline with the nf-core community early, as their contribution
   guidance recommends, rather than presenting a finished pipeline for transfer
   ([new pipeline guidance](https://nf-co.re/docs/contributing/contribute-new-pipelines/)).
7. After stable stage I/O exists, replace the monolithic local module with a
   subworkflow of stage modules where that improves resource placement.
8. Contribute a reusable smftools module to `nf-core/modules` only after its
   input/output and container contracts are stable. Multi-tool orchestration is
   generally better kept local to a pipeline first; nf-core modules must
   declare inputs/outputs, software versions, tests, and stubs
   ([module guidelines](https://nf-co.re/docs/guidelines/components/modules)).

The smftools license is MIT, its CLI is versioned, and it already has substantial
automated acceptance coverage, all of which are favorable. The missing
production container, workflow schema/samplesheet, nf-test coverage, stable
result JSON, and isolated stage-output contract are the main readiness gaps.

### Snakemake compatibility path

If Snakemake support is desired, use the same container and CLI contract:

- one rule per experiment calling `smftools experiment full` initially;
- config/sample wildcards rendered into a task-local experiment config;
- the complete experiment directory plus manifests as rule outputs;
- per-rule Apptainer or Conda software deployment; and
- later stage or partition rules only after immutable read-only I/O exists.

There is little value in maintaining equivalent full Nextflow and Snakemake
workflow implementations before the interface stabilizes. Maintain one
Nextflow reference because of nf-core, and test engine neutrality with a small
Snakemake wrapper or documented example.

### External-orchestration acceptance criteria

- A containerized BAM-entry experiment runs in an isolated directory with no
  undeclared host tools or writes.
- Requested `task.cpus`/memory/GPU map to smftools effective resources and are
  recorded.
- Every emitted output is declared and every required output is validated.
- A failed smftools stage makes the workflow process fail nonzero.
- `-resume` reuses an unchanged experiment process.
- Input, parameter, process-script, or container changes invalidate the
  external cache as expected.
- An old experiment upgrade reads the old directory without modifying it and
  emits a separately validated generation/run.
- Relocating published outputs preserves all internal relative links.
- Concurrent experiments never share a writable run root or current pointer.
- Container/version outputs include smftools and every invoked external tool.
- nf-test covers normal, failure, resume, and optional-stage paths.
- `nf-core pipelines lint` passes before any nf-core publication proposal.
- Docker and Apptainer profiles both complete the tiny test dataset; cloud
  object-storage execution is tested before claiming cloud portability.

## Current pipeline boundaries

### What “multi-reference alignment” currently means

There are three different operations that can easily be conflated:

1. **Reads aligned against the alignment FASTA.** `load_adata_core` calls the
   configured aligner and sorts the BAM before raw ingestion
   (`src/smftools/cli/load_adata.py:614-628`).
2. **Primary-alignment rescue across alternative FASTA records.** The optional
   rescue pass compares a read's primary and secondary alignments and promotes a
   better-covered alternative reference before any raw record commits the
   read's `Reference_strand`
   (`src/smftools/cli/load_adata.py:630-673`;
   `src/smftools/informatics/alignment_rescue.py:1-20`). Supplementary
   alignments are deliberately untouched.
3. **Reference-sequence alignment for variant annotation.** The current variant
   command globally aligns two reference sequences to each other, records
   substitutions/insertions/deletions, and builds a substitution coordinate map
   (`src/smftools/preprocessing/append_sequence_mismatch_annotations.py:25-161`).

Only the third operation belongs inside preprocessing. The first two define the
raw molecule's alignment and must precede raw extraction. Moving BAM alignment
or rescue after preprocessing begins would make `Reference_strand`, CIGAR,
mapped length, mismatch encoding, and every downstream position-dependent
structure internally inconsistent.

### Raw data already available to preprocessing

The partitioned raw store persists:

- the read sequence, base qualities, mismatch calls, modification signal, CIGAR,
  reference, and reference start in ragged Parquet;
- raw read/mapping/CIGAR/deamination metrics on the molecule spine; and
- original reference sequences plus encoding maps in spine `uns`.

On materialization, ragged arrays become `sequence_integer_encoding`,
`mismatch_integer_encoding`, `base_quality_scores`, and `read_span_mask`
(`src/smftools/informatics/ragged_store.py:509-538`). Deaminase strand-switch
metrics are computed during raw extraction
(`src/smftools/informatics/ragged_store.py:310-395`) and retained as scalar
spine columns (`src/smftools/informatics/raw_store.py:69-90`).

This is sufficient for early variant calls. One representation mismatch must be
resolved: partitioned materialization carries original sequences in
`uns["References"]`, but it does not reconstruct the legacy
`var["*_strand_FASTA_base"]` columns expected by the current variant functions.
The migrated implementation should consume a normalized reference catalog or
the stored sequences directly, rather than manufacturing a monolithic legacy
`var` layout merely to satisfy the old API.

## Current standalone variant stage

The legacy variant core currently performs the following:

1. optional sample-sheet enrichment and display-coordinate reindexing;
2. global alignment of exactly two configured reference-base columns;
3. per-reference mismatch-frequency and variable-site annotations;
4. per-read calls at reference substitution sites;
5. per-read segmentation, breakpoints, self/other counts, and chimera labels;
6. mismatch/sequence/segment plots, including several UMI annotation strips; and
7. writing a separate variant H5AD.

The principal outputs are documented in
`docs/source/tutorials/cli_usage.md:105-129` and implemented in
`src/smftools/preprocessing/append_variant_call_layer.py`.

### Existing variant output semantics

| Output | Meaning | Important limitation |
|---|---|---|
| `*_variant_call` | `1=seq1`, `2=seq2`, `0=unknown`, `-1=uninformative` | Calls only reference **substitutions**. Reference-alignment insertions and deletions are annotated but not called per read. |
| `*_variant_segments` | Dense span segmentation into seq1/seq2/transition states | One informative call can label a large span because the first/last class is extended to the read boundaries. |
| `*_breakpoint_count` / `*_is_chimeric` | Number/presence of class transitions | This detects a within-read switch, not merely disagreement with the assigned reference. |
| `chimeric_variant_sites` | Any segment classified as the reference other than the read's assigned reference | A fully or edge-discordant read can be true without a breakpoint. This may reflect chimera, misalignment, contamination, or sparse/noisy evidence. |
| `chimeric_variant_sites_type` | left/right/middle/multi-segment mismatch category | It is a geometric classification, not a calibrated probability or minimum-evidence QC decision. |
| `variant_self_base_count` / `variant_other_base_count` | Number of span bases filled as self/other | Counts are based on interpolated segments, not the number of informative variant observations. |

The distinction between transition-based and disagreement-based chimera flags is
visible in `append_variant_segment_layer`: transition counts are assigned at
`src/smftools/preprocessing/append_variant_call_layer.py:381-432`, while
`chimeric_variant_sites` is assigned for any other-reference segment at
`src/smftools/preprocessing/append_variant_call_layer.py:434-536`.

### Current stage defects relevant to migration

#### Critical: the standalone command does not support the partitioned pipeline

`variant_adata` resolves only legacy monolithic paths with a minimum stage of
`pp` (`src/smftools/cli/variant_adata.py:47-55`). `AdataPaths` has no
partitioned variant spine, and the partitioned-stage required-artifact map has no
variant entry (`src/smftools/constants.py:108-150`). A default partitioned
preprocess run can therefore complete successfully while leaving no monolithic
`pp` or `pp_dedup` H5AD for the variant command to consume.

The repository contains standard plot-layout categories for a future
partitioned variant stage (`src/smftools/cli/stage_artifacts.py:21-38`), but no
executor, catalog, read index, spine publication, or lifecycle integration uses
them.

#### High: variant runs after filtering and duplicate removal

The variant path name is derived from `pp_dedup`, and its minimum input is
preprocessed (`src/smftools/cli/helpers.py:393-400`;
`src/smftools/cli/variant_adata.py:47-55`). It therefore cannot:

- annotate reads that failed earlier QC;
- drive a variant-aware QC mask;
- measure variant events before duplicate removal; or
- inspect whether a duplicate cluster contains variant-discordant members.

This directly conflicts with the desired behavior.

#### High: the full workflow omits variant analysis

After PL-23, `full_flow` executes raw, preprocess, spatial, HMM, and latent by
default, with latent configurable as an opt-out
(`src/smftools/cli/recipes.py:117-153`). It still does not invoke the standalone
variant command. Consequently, downstream options such as
`omit_chimeric_reads` or variant overlays do not reliably have variant
annotations in the normal pipeline. Legacy HMM contains a late best-effort
backfill from a separately written variant H5AD
(`src/smftools/cli/hmm_adata.py:1350-1365`), while partitioned downstream
stages have no analogous source.

#### High: redo and compatibility controls are incomplete

`variant_adata` checks `getattr(cfg, "force_redo_variant_analyses", False)`, but
`ExperimentConfig` and `default.yaml` do not define or construct that field. The
core also writes only when the variant path does not exist, so even a dynamically
injected force flag would not replace the H5AD
(`src/smftools/cli/variant_adata.py:40-44`, `705-715`).

Plot skipping inconsistently uses `force_redo_preprocessing`, directory
existence, or unconditional existence checks. Variant recomputation is therefore
not governed by a coherent stage-level contract.

#### High: “multi-reference” support is actually one reference pair

The config accepts a two-element list of legacy `var` column names
(`src/smftools/config/experiment_config.py:1066-1074`), and the core unpacks
exactly two values (`src/smftools/cli/variant_adata.py:172-176`). Output prefixes
can distinguish a pair, but completion flags such as
`append_sequence_mismatch_annotations_performed`,
`append_variant_call_layer_performed`, and
`append_variant_segment_layer_performed` are global booleans. A second pair in
the same object would be skipped unless every caller supplied pair-specific
flags and force behavior.

The reference-to-sequence mapping also relies on stripping
`"_strand_FASTA_base"` from column names and matching the result exactly to
`Reference_strand`
(`src/smftools/preprocessing/append_variant_call_layer.py:140-162`). This is
fragile for arbitrary named reference sets, more than two alleles, or aliases.

#### Medium: population mismatch-frequency analysis and known-reference calling
are mixed together

`append_mismatch_frequency_sites` derives site frequencies from the currently
present reads and optionally compares them to mean base-call error probability
(`src/smftools/preprocessing/append_mismatch_frequency_sites.py:88-187`).
Reference-pair variant calling instead uses known substitutions from a global
reference-to-reference alignment. These products have different dependency and
QC semantics:

- known-reference informative sites are deterministic given the reference set
  and algorithm version;
- mismatch frequencies depend on the selected read cohort; and
- using a cohort-derived frequency to filter the same cohort can create a
  circular or quality-sensitive decision.

They should be separate analysis nodes with separate cohort provenance.

#### Medium: legacy preprocessing discards variant inputs early

The legacy preprocess core intentionally removes
`sequence_integer_encoding` and `mismatch_integer_encoding` near its start and
reattaches them only before saving
(`src/smftools/cli/preprocess_adata.py:336-345`, `725-746`). An early variant
step must run before this memory optimization or retain only the minimum
variant-event representation needed by later QC.

#### Medium: tests cover components, not the desired integrated contract

There are focused unit tests for reference mismatch annotation, mismatch
frequencies, segmentation labels, plotting, and selected partitioned preprocess
behaviors. There is no real end-to-end test that:

- runs variant calling through partitioned preprocessing;
- handles multiple reference pairs/sets;
- validates variant QC masks;
- compares variant metrics before and after dedup;
- validates genome tile-boundary behavior; or
- upgrades a prior preprocess generation with a newly added analysis.

The targeted existing tests run during this audit passed (`9 passed`, with 19
tests deselected by the audit's focused `-k` expression), but they establish
component stability rather than migration readiness.

## Current preprocessing and QC order

### Legacy path

The legacy path physically subsets the AnnData:

```text
sample/UMI metadata
  -> length/quality/mapping filter
  -> CIGAR-indel filter
  -> binarization/NaN transforms
  -> coverage/base context/read modification metrics
  -> modification filter
  -> recompute coverage/context
  -> deaminase chimera label (label only)
  -> duplicate detection
  -> save QC-filtered full object
  -> save deduplicated subset
```

Evidence is in `src/smftools/cli/preprocess_adata.py:347-724`. Importantly,
legacy deaminase chimera labeling does not remove reads
(`src/smftools/cli/preprocess_adata.py:657-665`).

The saved non-deduplicated `pp` object is already missing reads removed by read,
CIGAR, and modification QC. It is not a pre-QC archive.

### Partitioned path

The partitioned path retains all raw spine rows and computes masks:

```text
all raw rows
  -> passes_read_qc (length/quality/mapping + CIGAR)
  -> deaminase_PCR_chimera label and exclusion from passes_read_qc
  -> task-local transforms and reduced modification metrics
  -> passes_modification_qc
  -> passes_qc = passes_read_qc & passes_modification_qc
  -> duplicate clustering among passes_qc reads
  -> passes_dedup = passes_qc & ~is_duplicate
```

This contract is documented and implemented at
`src/smftools/preprocessing/partitioned_executor.py:477-582`,
`699-728`, and `731-852`.

There is a behavior discrepancy: the schema registry and legacy path describe
`deaminase_PCR_chimera` as label-only, but partitioned preprocessing removes
those reads from `passes_read_qc`
(`src/smftools/schema/anndata_schema_v1.yaml:196-213`;
`src/smftools/preprocessing/partitioned_executor.py:555-577`). Variant migration
should not compound this ambiguity; mask semantics need to be made explicit and
tested across both modes.

## Recommended target preprocessing order

The recommended sequence is:

```text
raw alignment + optional secondary-alignment rescue
  -> raw partitioned store and molecule spine
  -> sample/UMI metadata needed for grouping
  -> reference-set planning and reference-to-reference alignment
  -> per-read variant calls and evidence metrics on every raw molecule
  -> baseline technical read QC
  -> modification transforms, metrics, and modification QC
  -> variant QC mask
  -> duplicate clustering/reconciliation
  -> named cohort summaries (including pre/post dedup variant metrics)
  -> publish preprocess generation
  -> plots as derived artifacts of the published cohorts
```

### Split variant work into three analysis nodes

#### 1. Reference-set definition and informative-site catalog

Create a normalized, versioned reference catalog from the stored original
sequences. For each configured pair or allele set it should record:

- stable reference-set and member IDs;
- reference identities/checksums and orientation;
- the alignment algorithm and scoring/version;
- substitution and indel coordinate mappings;
- conversion-aware acceptable base sets;
- informative/uninformative status and reason; and
- stored/original coordinate mappings.

This is experiment/reference-set scope, not read-task scope. It should run once
per unique compatible reference set and be reusable by every read task.

The current two-column config should remain a backward-compatible translation,
but the target user config should identify references or named allele sets
directly. It should not expose legacy `var` column names as the primary public
contract.

#### 2. Per-read variant evidence

Run before other QC on every raw molecule. At minimum, persist:

- informative sites covered;
- informative sites with a recognized allele call;
- unknown/other calls;
- calls for each allele/reference member;
- callable fraction;
- transition/breakpoint count and positions;
- self/other evidence counts at informative sites;
- interpolated self/other span counts, if still useful;
- classification and classification reason; and
- the reference-set/catalog version used.

Counts at genuinely informative sites must remain distinct from interpolated
segment lengths. QC should primarily use evidence counts/fractions; dense
segment layers are better treated as visualization products.

For large genomes, a sparse/event representation is preferable to adding
another experiment-wide dense int8 layer. Per-read/per-site calls can be stored
as Parquet or task-local arrays and materialized into a plotting window when
requested.

#### 3. Cohort-dependent mismatch frequencies and summaries

Compute mismatch frequency separately for named cohorts. At least:

- `all_aligned`;
- `passes_read_qc`;
- `passes_nonvariant_qc` (technical plus modification QC);
- `passes_variant_qc`;
- `pre_dedup` under the explicitly selected final QC policy; and
- `post_dedup`.

Every result row should include the cohort expression or mask-version ID. This
avoids silently treating a frequency derived after filtering as if it described
the raw read population.

### Genome/tile correctness

The current variant functions assume a complete dense locus. A direct call from
each existing preprocess position task would be incorrect for genome-mode reads
that cross core boundaries:

- the same read can occur in multiple position cores;
- a transition can occur between informative sites owned by adjacent cores;
- per-read outputs would be duplicated or incomplete; and
- a dense segment extending to the read boundary cannot be finalized from one
  partial core.

Two safe designs are available:

1. use a molecule-owned variant task keyed by
   reference/reference-set + barcode/read chunk, loading each read's full
   aligned span and writing exactly one per-read result; then separately write
   core-owned per-position overlays; or
2. emit ordered per-core informative-call partials and reduce them by molecule
   before calculating breakpoints and classifications.

The first is simpler for locus experiments. The second better reuses existing
genome tiling but requires a well-tested reducer. In either case, exactly one
artifact must own each final per-molecule metric.

## Variant-aware QC recommendations

### Preserve independent masks

Do not encode filtering only by physical row removal. Add:

- `passes_read_qc`;
- `passes_modification_qc`;
- `passes_variant_qc`;
- `passes_nonvariant_qc`;
- `passes_qc`;
- `is_duplicate`; and
- `passes_dedup`.

Also persist categorical/list-like reason fields, for example
`variant_qc_reason`, rather than only a boolean. This makes threshold changes,
audits, and before/after summaries possible.

### Candidate variant QC metrics

Reasonable configurable metrics include:

- minimum informative sites covered;
- minimum recognized-call count or callable fraction;
- maximum unknown/ambiguous-call fraction;
- maximum other-reference informative-call fraction;
- allowed breakpoint count/types;
- minimum supporting calls on each side of a breakpoint;
- minimum segment size in informative sites and bases; and
- optionally exclusion of specific calibrated chimera classes.

The existing `chimeric_variant_sites` boolean should not be the sole default
filter. It has no minimum supporting-site threshold and conflates several
biological/technical explanations. Initially, it is safer to make the label and
its evidence available, default variant QC to permissive, and validate stricter
defaults on representative data.

### Interaction with duplicate detection

If variant QC failures are removed before duplicate clustering, they cannot be
assigned a duplicate cluster and “before versus after dedup” chimera metrics are
not well-defined. If duplicate clustering ignores variant status entirely, a
variant-failing read can become the keeper while a variant-passing cluster member
is removed.

A robust policy is:

1. compute clusters on reads passing non-variant QC;
2. prefer variant-QC-passing members when choosing the representative;
3. retain cluster assignments for all clustered reads;
4. define final `passes_dedup` from non-variant QC, variant QC, and keeper status;
   and
5. report within-cluster variant discordance.

This preserves event accounting and avoids losing an otherwise valid cluster
solely because the current `read_quality` keeper happened to fail variant QC.
Changing keeper policy is user-visible and must be explicit/configured.

## Before/after duplicate metrics

### Required denominator discipline

“Before duplicate filtering” and “after duplicate filtering” must identify their
other QC conditions. Recommended labels are:

- `pre_dedup_nonvariant_qc`: `passes_nonvariant_qc`;
- `post_dedup_nonvariant_qc`: one selected representative per duplicate cluster
  among `passes_nonvariant_qc`;
- `pre_dedup_final_qc`: `passes_qc`;
- `post_dedup_final_qc`: `passes_dedup`; and
- optionally `all_aligned` for raw prevalence.

This prevents a variant filter from making its own event rate appear to be zero
by definition.

### Recommended variant-derived summary rows

For experiment, sample, assigned reference, reference set, and optional region:

- total reads and callable reads;
- reads with any other-reference evidence;
- reads with one or more breakpoints;
- counts/rates by `chimeric_variant_sites_type`;
- breakpoint-count distribution;
- self/other informative-call counts and fractions;
- interpolated self/other span fractions, separately labeled;
- number of duplicate clusters containing a variant-resolved event;
- number of mixed-status clusters;
- fraction of event-positive reads marked duplicate;
- fraction of event-positive clusters retained after dedup; and
- absolute and relative pre/post-dedup change.

A long-form `variant_qc_metrics.parquet` with numerator, denominator, value,
cohort, grouping columns, analysis version, and source generation is preferable
to embedding only summary dicts in H5AD `uns`.

Plots should be generated from this table and from bounded materializations, not
be the only durable representation of the metric.

## Incremental application to previous outputs

### What is possible today

#### Legacy H5AD

Legacy H5AD output records a `smftools.history` entry containing package version,
resolved parameters, input records, output keys, runtime, and status
(`src/smftools/metadata.py:346-443`). Runtime schema snapshots also describe
present keys.

Individual functions usually use one boolean `uns` flag such as
`append_variant_call_layer_performed`. Those flags answer only “was some version
of this function reported complete?” They do not establish:

- which parameter values were used;
- which reference pair was processed;
- which input generation was consumed;
- which algorithm implementation produced the output;
- whether the output keys are complete/readable; or
- whether a changed function needs to be rerun.

Top-level skip behavior is based largely on output path existence. If
`pp_dedup` exists, preprocessing returns without inspecting whether a new
analysis is missing (`src/smftools/cli/preprocess_adata.py:222-229`).

`force_redo_preprocessing` also prefers an existing `pp` object over raw input
(`src/smftools/cli/preprocess_adata.py:132-158`). Since `pp` is already
QC-filtered, a rerun cannot restore reads removed by the old thresholds. It can
backfill an additive analysis only for retained reads.

Therefore:

- a raw legacy H5AD can be fully reprocessed;
- a non-deduplicated `pp` H5AD can receive additive analyses for its retained
  rows and can recompute dedup, but cannot recover earlier QC failures;
- a deduplicated H5AD can describe only surviving representatives; and
- prior boolean flags/history can support a best-effort migration report, not a
  proof of semantic compatibility.

#### Partitioned output

Partitioned output is a much stronger starting point:

- raw ragged data remains authoritative;
- the preprocess spine preserves all molecule rows and masks;
- deterministic task catalogs and derived read indexes locate task outputs;
- the experiment manifest records stage config hashes, artifact paths, task
  counts, and schema versions; and
- the consolidated experiment spine can union stage pointers and normalized
  `obs` sidecars.

This allows a new analysis to be computed for all original reads without
re-running basecalling/alignment, provided the raw store contains every input
channel required by the new analysis.

### Why current partitioned skip logic is insufficient

The stage lifecycle compares a semantic config hash and validates required
artifacts (`src/smftools/cli/helpers.py:171-207`, `291-369`;
`src/smftools/informatics/experiment_manifest.py:228-292`). This is useful but
still stage-granular.

Current gaps include:

- preprocess has no per-analysis registry or dependency graph;
- the preprocess config hash currently includes nearly every resolved experiment
  config value except resource/force/plot exclusions, so unrelated downstream
  config changes can invalidate preprocessing;
- the hash does not include an explicit preprocessing algorithm version;
- `smftools` version and git commit are not part of stage compatibility;
- `schema_versions={"preprocess": 2, ...}` is recorded on publication, but the
  preprocess skip call does not request an expected schema match;
- the normal preprocess skip calls `partitioned_stage_is_complete` without a
  source path, so it does not compare recorded input artifact identities;
- task paths are deterministic canonical paths, not immutable generation paths;
- `prepare_derived_read_index` deletes the prior index before a rerun
  (`src/smftools/informatics/derived_read_index.py:25-31`);
- every preprocess task is dispatched again; existing task results are not
  classified/reused individually; and
- the executor publishes one preprocess snapshot rather than appending an
  independent analysis generation
  (`src/smftools/preprocessing/partitioned_executor.py:962-1143`).

PL-17's implemented latent lifecycle demonstrates the required
pattern—source/config compatibility, immutable generation directories,
transactional publication, and compute-generation reuse—but that pattern has
not yet been generalized to preprocessing or analysis-level reuse. The
generalization should use the same shared experiment-manifest lifecycle rather
than create a second cache authority.

## Recommended incremental analysis contract

### Analysis registry

Define each preprocessing operation as a declarative node with:

- stable `analysis_id`, for example `variant.reference_catalog`,
  `variant.read_calls`, `qc.variant_mask`, `dedup.sequence_clusters`, or
  `metrics.variant_by_cohort`;
- semantic `algorithm_version`, incremented only when results may change;
- output `schema_version`;
- exact semantic config keys;
- required input channels and schemas;
- upstream analysis dependencies;
- execution scope: experiment, reference set, molecule chunk, position core, or
  reducer;
- output artifacts and ownership rules; and
- downstream invalidation edges.

Record the installed package version, git commit, Python/dependency environment,
and timestamps for provenance, but do not automatically invalidate every
analysis after any package upgrade. Compatibility should depend on the explicit
per-analysis algorithm version plus config/input identities.

### Compatibility key

For an analysis/task, use a key equivalent to:

```text
analysis_id
+ algorithm_version
+ output_schema_version
+ semantic_config_hash
+ ordered input artifact IDs/checksums
+ dependency generation IDs
+ task logical identity
```

Presence of expected columns/layers is a validation check, not the compatibility
key by itself.

### Planner outcomes

An upgrade planner should report one of:

- `compatible`: validated artifact already exists;
- `missing`: analysis is new and may be appended;
- `stale_config`: semantic parameters changed;
- `stale_algorithm`: algorithm version changed;
- `stale_input`: raw/reference/dependency generation changed;
- `invalid_artifact`: manifest says complete but output validation fails;
- `blocked_missing_input`: an old store lacks a required raw channel; or
- `dependent_recompute`: an upstream mask/result changed.

A dry-run/plan view should show why each node will run or skip before mutation.

### Append versus recompute rules

Safe analysis-reuse examples within a newly published generation:

- a new diagnostic per-read metric that consumes immutable raw channels;
- a new cohort summary over unchanged masks;
- a new plot over an unchanged compatible analysis generation; or
- variant annotations added in reporting-only mode.

Not purely append-only:

- enabling variant-based exclusion, because `passes_qc` changes;
- changing duplicate candidates or keeper preference, because dedup changes;
- changing reference assignment/alignment rescue, because raw molecule
  coordinates change;
- changing informative reference sets, because variant calls change; or
- changing a mask consumed by spatial/HMM/latent planning.

All cases publish a new preprocess generation; the distinction is how much
compute can be reused. The latter cases must also mark dependent stage
generations incompatible. Old generations should remain inspectable until an
explicit cleanup policy removes them.

### Artifact layout

A practical layout, deliberately matching PL-17, is:

```text
preprocess_adata_outputs/
  current.json
  generations/<preprocess_generation_id>/
    generation_manifest.json
    spine.h5ad
    analyses/
      variant.reference_catalog/<analysis_generation_id>/...
      variant.read_calls/<analysis_generation_id>/...
      qc.variant_mask/<analysis_generation_id>/obs.parquet
      dedup.sequence_clusters/<analysis_generation_id>/obs.parquet
      metrics.variant_by_cohort/<analysis_generation_id>/metrics.parquet
    task_catalog.parquet
    read_index/
    plots/
```

The executor should build this under `.staging/<preprocess_generation_id>`,
validate it, move it atomically into `generations/`, publish the canonical
preprocess spine and `current.json`, and only then complete the stage manifest
and consolidated experiment spine. The canonical preprocess spine is a
pointer/view over one validated generation, not the mutable location where task
writers work. Analysis subdirectories are components referenced by the
generation manifest; they are not separately mutable “current” stores.

### Legacy import policy

For old monolithic H5ADs:

1. inspect `smftools.history`, runtime schema, output keys, and legacy flags;
2. classify imported analysis provenance as `known`, `inferred`, or `unknown`;
3. never equate a bare `*_performed=True` with the new compatibility key;
4. append only analyses whose required inputs are present;
5. clearly state when results cover only the old retained/deduplicated row set;
   and
6. require raw input for any upgrade that needs previously filtered reads or
   altered alignment/reference assignment.

## Migration strategy

### Phase 1: establish semantics without changing defaults

- Add reference-set and per-read variant analyses to preprocessing in
  reporting-only mode.
- Run before read QC on every raw molecule.
- Keep `passes_variant_qc=True` by default.
- Publish evidence metrics, cohort tables, and pre/post-dedup summaries.
- Validate parity with the current legacy variant outputs on representative
  locus data.

### Phase 2: partitioned-native execution and plots

- Implement experiment/reference-set catalog planning.
- Add molecule-owned calls/reducers that are correct across genome cores.
- Store sparse/event outputs plus optional task-local overlays.
- Add normalized variant `obs` and metric sidecars to the preprocess generation.
- Generate plots from named cohorts and register them in the preprocess plot
  catalog.

### Phase 3: configurable variant QC

- Add validated config fields for evidence thresholds and allowed classes.
- Preserve independent masks/reasons.
- Define duplicate candidate and keeper interaction explicitly.
- Keep legacy and partitioned mask semantics identical.
- Add migration notes for any default filtering behavior.

### Phase 4: incremental analysis reuse within preprocess generations

- Extend the PL-17 stage lifecycle with the analysis registry, compatibility
  keys, dry-run planner, per-node reuse, and downstream invalidation.
- Always publish a new immutable preprocess generation; never append into the
  current generation directory.
- Backfill new analyses from partitioned raw stores without rerunning unchanged
  transforms.
- Treat legacy H5AD migration as best-effort with row-set limitations.

### Phase 5: retire the standalone variant branch

For backward compatibility, keep `smftools experiment variant` temporarily as a
thin alias that requests the variant analysis/plot nodes from preprocessing and
prints a deprecation notice. Remove separate variant H5AD stage resolution only
after documented migration support exists.

The standard `full` workflow should then receive variant annotations naturally
through preprocess, allowing spatial/HMM filters and overlays to consume one
authoritative preprocess generation.

## Required tests and acceptance criteria

### Variant computation

- Known two-reference legacy fixtures produce equivalent informative
  substitutions and read calls after migration.
- Conversion acceptable-base sets remain disjoint/calibrated as intended.
- Insertions/deletions are either explicitly supported per read or clearly
  excluded with tested semantics.
- Multiple reference pairs/allele sets do not collide in flags, columns, or
  artifacts.
- Sparse/no-call, fully discordant, one-breakpoint, middle-segment, edge-segment,
  and multi-segment reads have explicit expected classifications.
- Minimum-evidence thresholds prevent one isolated call from becoming an
  unintended hard-filter event.

### Partitioning

- Locus and genome outputs are invariant to task memory size, barcode chunking,
  worker count, and task order.
- A read crossing adjacent cores receives exactly one final per-read metric row.
- Breakpoints spanning a core boundary are detected.
- Query/materialization retrieves variant outputs without opening unrelated task
  stores.

### QC and dedup

- All mode combinations produce the same named masks in legacy and partitioned
  paths.
- Variant failure reasons are retained even when final reads are excluded.
- Keeper selection prefers a variant-pass member when configured.
- Mixed-status duplicate clusters are counted and reproducible.
- Pre/post-dedup metrics have tested numerators and denominators.
- Deaminase label-only versus exclusion behavior is reconciled and documented.

### Incremental upgrades

- A new independent analysis publishes a new immutable preprocess generation
  without rerunning compatible preprocess tasks or mutating the old generation.
- A plot-only change reuses compute artifacts.
- A semantic variant config change reruns variant and dependent nodes only.
- An algorithm-version change invalidates its analysis even with unchanged
  config.
- A QC-mask change creates a new dedup generation and invalidates dependent
  stages.
- A changed raw/reference source cannot be mistaken for a cache hit.
- Missing/corrupt task output fails compatibility validation.
- Fault injection never publishes a partial generation as current.
- Old generations remain readable after a failed replacement.
- A legacy deduplicated H5AD upgrade reports that removed reads are unavailable.

## Decisions needed before implementation

1. Is the intended primary variant model still a pair of known reference
   sequences, or must one experiment support more than two alleles/reference
   members?
2. Should `chimeric_variant_sites` remain a broad diagnostic, with a new stricter
   QC classification, or should its existing semantics change?
3. Which variant metrics should be hard filters by default, if any?
4. Should duplicate clustering include variant-failing reads for cluster/event
   accounting, with keeper preference for passing reads?
5. Must per-read reference indels be called, or is substitution-only calling an
   accepted first contract?
6. For legacy outputs, is a retained-read-only backfill acceptable, or should
   upgrades require/recover the original raw artifact?
7. Should the first external workflow accept aligned, modification-bearing BAM
   input only, or must its first release also own POD5/FAST5 basecalling,
   demultiplexing, alignment, and modification extraction?
8. Is one containerized `experiment full` process per experiment sufficient for
   the first Nextflow release, or is cluster-scale partition scatter/gather an
   immediate requirement?
9. Is the initial publication target an independent Nextflow pipeline, or is
   nf-core adoption near-term enough to require its samplesheet, schema,
   testing, container, and community conventions from the first prototype?

## Conclusion

The requested migration is possible and fits the package's partitioned
direction. The key is not to paste the existing `variant_adata_core` into the
top of `preprocess_adata_core`. The current implementation assumes a complete
legacy dense object, exactly one reference pair, global completion booleans, and
post-dedup inputs. A production-grade migration should separate immutable
reference-set planning, molecule-owned variant evidence, cohort-dependent
summaries, QC masks, and plots.

The partitioned raw store already provides the strongest prerequisite for both
variant-aware QC and future upgrades: all molecules and raw sequence channels
remain available after preprocessing. Adding analysis-level provenance,
per-node reuse inside PL-17-style immutable generations, and dependency-aware
compatibility would make it possible to apply newly introduced preprocessing
analyses to prior partitioned experiments without rerunning unrelated work,
while still forcing new QC/dedup/downstream generations when scientific results
would change. This extends the companion project/latent plan rather than
creating a competing lifecycle, identity, or cache model.

The package is already a credible engine-neutral scientific core and is
amenable to a coarse-grained Nextflow or Snakemake wrapper. The recommended
first external workflow is one isolated `experiment full` process per
experiment, beginning from aligned BAM input, while smftools retains its
internal bounded executors. For eventual nf-core publication, prioritize a
pinned production container, a stable structured input/result contract,
read-only process inputs, compact QC/version artifacts, and workflow-level
tests. Only split stages or partitions into external DAG nodes after their
outputs become immutable self-contained bundles; do not duplicate
Nextflow/Snakemake scheduling inside the package.

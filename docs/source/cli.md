# Command-line interface

```{click} smftools.cli_entry:cli
:prog: smftools
:nested: full
```

## Read-only project planning

Use `smftools project plan PROJECT_DIR TARGET CANONICAL_REFERENCE` to inspect a
project analysis dependency plan without publishing artifacts or changing the
project registry. Add `--json` for deterministic machine-readable output; the
other selection and projection options mirror project materialization.

Each plan target maps to one execution and validation lifecycle:

| Plan target | Execute with | Task-local artifact | Validate with |
| --- | --- | --- | --- |
| `selection` | *(not executable)* | none -- it is the membership/feature dependency the other three consume | -- |
| `materialization` | `project run` (default), or `project materialize` for a non-task-local pool | `materialized.h5ad.gz`, or a partitioned directory | `project validate` |
| `sample-analysis` | `project run --target sample-analysis`, or `project sample-analysis` | `sample_analysis.parquet` | `project validate` |
| `embedding` | `project run --target embedding`, or `project embedding` | `embedding.parquet` | `project validate` |

`project run` is the engine-facing entry point for every executable target, the
same way `experiment run --target` is for experiment stages; the named
subcommands remain for interactive use. `run` accepts the union of the targets'
options and rejects any that do not apply to the chosen target rather than
ignoring them, since a silently dropped flag would publish a result that does not
describe what was requested.

Use `smftools experiment plan CONFIG --target TARGET` for the corresponding
read-only experiment plan. The compatibility states, force behavior, and
semantic-versus-execution distinction are described in
[](tutorials/semantic_variant_workflows.md).

Add `--upgrade-impact` to either plan command to group the same compatibility
decisions by `PlanState`, separate triggering nodes from downstream dependent
recomputation, and summarize recompute cost. This mode is a read-only projection
over the existing planner; it does not add or relax invalidation rules:

```shell
smftools experiment plan experiment.csv --target full --upgrade-impact
smftools project plan PROJECT_DIR embedding REFERENCE_UID --upgrade-impact --json
```

Experiment cost estimates sum previously observed `elapsed_seconds` values from
the experiment manifest. Coverage is explicit: nodes without a prior timing are
listed as unknown and any numeric total is marked partial. Project products are
task-local and project cache definitions are more detailed than the coarse plan
arguments, so project impact reports cost as unknown rather than guessing from
artifact size or treating a cache as a generic compatible result. The JSON
report is schema-versioned independently from the source semantic plan.

## Named experiment sets

`--set NAME` restricts a project command to a saved subset of the registered
experiments. Manage those subsets with `project add-set`, `project list-sets`,
`project show-set`, and `project remove-set`; membership is validated when the
set is defined, and `show-set` resolves through the same path `--set` applies,
so what it prints is what a plan or materialize will use. See
[](tutorials/cli_usage.md#smftools-project).

## Renaming an experiment

Use `smftools experiment rename-id` to change a human-readable experiment ID
without changing its durable `experiment_uid`:

```shell
smftools experiment rename-id /data/runs/old-id new-id \
  --project /data/projects/project-a \
  --project /data/projects/project-b
```

The command performs complete collision and identity preflight before writing,
then transactionally updates the experiment manifest, the standard
`experiment_config.csv` when present, every explicitly supplied project
registry, explicit-list named sets, and project per-sample pointers. It moves
the experiment directory and per-sample state to the new ID and rolls the
completed writes and moves back if publication fails. A durable journal also
restores a prepared transaction before a retry after process interruption.
Supply `--config PATH` for a config stored outside the experiment directory.

Projects are not globally discoverable, so repeat `--project` for every project
that registers the experiment. Query-defined named sets are stored SQL and are
left unchanged and reported for review; update a query explicitly if its SQL
names the old ID. Published stage generations and historical provenance are
immutable and are not rewritten.

## Generation inventory and retention

Use `smftools experiment generations OUTPUT_ROOT` to inventory every immutable
generation below one experiment. The current generation is marked with `*`, an
explicitly retained generation is marked with `P`, and `--size` totals the bytes
inside each generation. `--json` emits inventory schema version 2, which adds
`pinned` and `retention_reasons` to each generation record.

Pins are mutable policy metadata, so they live in `retention.json` beside the
stage's `current.json`; published `generation_manifest.json` files remain
unchanged. A generation can have several independently removable reasons:

```shell
smftools experiment generations OUTPUT_ROOT pin raw GENERATION_ID \
  --reason "paper figure 3"
smftools experiment generations OUTPUT_ROOT pin raw GENERATION_ID \
  --reason "SRA:ABC123"
smftools experiment generations OUTPUT_ROOT unpin raw GENERATION_ID \
  --reason "paper figure 3"
smftools experiment generations OUTPUT_ROOT unpin raw GENERATION_ID \
  --all-reasons
```

Pruning is planning-only in EGL-03a. The command evaluates age and count policy,
walks generation sizes, and never deletes artifacts:

```shell
smftools experiment generations OUTPUT_ROOT prune \
  --keep-last 2 \
  --older-than 2026-01-01T00:00:00Z
```

Current, pinned, unreadable, recent, and newest retained generations are always
kept. Older policy matches are reported as `blocked_reproducibility`, with zero
reclaimable bytes, until retained inputs and provenance can prove byte-level
reproducibility. `--json` emits a versioned plan with `dry_run: true` and
`deletion_supported: false`; there is no delete or force mode in this phase.

## Project analysis cache inventory

Use `smftools project analyses list PROJECT_DIR` to inspect periodicity and
embedding caches without loading result tables, arrays, or persisted estimator
pickles. Each entry reports its project-relative path, analysis scope, size, and
stored versus installed algorithm and semantic-graph versions.

Entries are `current` when both code-identity versions match, `stale` when an
upgrade changed or introduced either version, and `invalid` when the cache's
definition or current-generation metadata is incomplete or unreadable. Legacy
caches without code-identity fields are therefore visible as stale rather than
being silently reused. Add `--stale` to show only stale or invalid entries, or
`--json` for the versioned machine-readable inventory. The command never rewrites
or removes caches.

## External workflow contract

Workflow engines should use `smftools experiment run` instead of rewriting an
experiment config or parsing stage logs:

```shell
smftools experiment run experiment.csv \
  --target full \
  --output-root "${TASK_OUTPUT}" \
  --input "${STAGED_BAM}" \
  --fasta "${STAGED_FASTA}" \
  --cpus "${TASK_CPUS}" \
  --memory-gb "${TASK_MEMORY_GB}" \
  --strict
```

The command writes a task-local runtime config, `software_versions.json`, and
`workflow_result.json` inside `--output-root`. The result records the semantic
plan, terminal outcome, generation and result IDs, relative artifact pointers,
available checksums, schemas, timings, structured failures, and the bounded
resource decision. Success, compatible reuse, and failure are represented by
the stable outcomes `success`, `compatible_skip`, and `failed`.

`--input` and `--fasta` accept concrete local files and local `file://` URIs.
Directory and remote URI inputs are not accepted in workflow mode; stage them
to one task-local file first. Read-only aliases are created inside the output
root so indexes and sidecars are also owned by the task. Overrides are applied
to the task-local config copy, and the source config and staged inputs are
integrity-checked without being rewritten. CPU and memory overrides can only
reduce the resolved config/host envelope. A requested CUDA or MPS accelerator
must also be available, and a CPU-only config cannot be expanded to an
accelerator.

Validate a completed or relocated bundle without writing:

```shell
smftools experiment validate "${TASK_OUTPUT}" --json
```

Validation exits nonzero for a failed result, incomplete or semantically
incompatible stage, missing/corrupt artifact, checksum mismatch, or pointer
that is absolute or escapes the output root. Internal workflow pointers are
relative to the output root, so moving the complete directory preserves the
contract.

Use `smftools versions --json` for the stable smftools/Python record. Repeat
`--tool` with a supported workflow executable (`dorado`, `pod5`, `minimap2`,
`modkit`, `gzip`, `multiqc`, `samtools`, `bedtools`, or
`bedGraphToBigWig`) to probe external versions explicitly. A workflow result
automatically records the tools required by the stages it is about to execute
and configured model identities; `--strict` fails before computation if one is
unavailable. In the [production CPU container](containers.md), the versions
record also includes the image, immutable tag, source revision, execution
profile, and runtime-supplied registry digest.

Project materialization uses the same result schema and task-local ownership:

```shell
smftools project run PROJECT_DIR CANONICAL_REFERENCE \
  --output-root "${TASK_OUTPUT}" \
  --layers C_site_binary \
  --cpus "${TASK_CPUS}" \
  --memory-gb "${TASK_MEMORY_GB}"

smftools project validate PROJECT_DIR "${TASK_OUTPUT}" --json
```

An unchanged project selection and projection request reuses a checksum-valid
materialization with the `compatible_skip` outcome. Project validation compares
the current semantic source plan with the published plan and reports stale
membership or feature inputs.

Project sample analysis is a separate product under the same contract, with its
own command name and artifact ID so neither can be skipped against the other:

```shell
smftools project sample-analysis PROJECT_DIR CANONICAL_REFERENCE \
  --output-root "${TASK_OUTPUT}" \
  --layer C_site_binary

smftools project validate PROJECT_DIR "${TASK_OUTPUT}" --json
```

`smftools project embedding` completes the set of executable plan targets. It
publishes the shared coordinate system as a project-scoped immutable generation
and exports the coordinates task-locally. Extending an existing embedding loads
this project's persisted estimator pickles, so it fails unless the run opts in
with `--trust-local-models`; that decision is recorded in the result.

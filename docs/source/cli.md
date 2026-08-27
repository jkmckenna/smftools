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

## Selective re-basecall planning

Use the nested re-basecall planner to inspect an immutable parent generation,
source-signal availability, and one structured molecule selection without
running Dorado or writing scientific artifacts:

```shell
smftools experiment rebasecall plan experiment.csv request.yaml
smftools experiment rebasecall plan experiment.csv request.yaml --json
```

A minimal QC request is:

```yaml
schema_version: 1
name: publication-cohort
source:
  raw_generation: current
  preprocess_generation: current
selection:
  mode: qc
  predicate:
    all:
      - {column: passes_read_qc, op: eq, value: true}
      - {column: passes_dedup, op: eq, value: true}
basecall:
  model: hac@latest
signal:
  materialize: false
downstream:
  target: full
promotion:
  activate: false
```

Schema-1 requests support `all-signal`, `all-parent-molecules`, `qc`, and `ids`
selection modes. QC predicates are bounded structured objects over canonical QC
mask columns; arbitrary Python and SQL expressions are rejected. Plans report
the exact raw and optional preprocess generation IDs, selection universe and
count, source-manifest identity and availability, scientific-scope warnings,
and stable blocking reason codes.

This first planning contract deliberately does not freeze a selection, validate
relocated POD5 checksums, resolve a floating Dorado model selector, execute a
basecall, or publish/promote a lineage. Those capabilities remain explicit in
the plan as the SRB-02 through SRB-05 delivery boundaries. A request must set
`promotion.activate: false`; later promotion is always a separate operation.

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

## Named roots

`smftools data roots list [--config-dir DIR] [--json]` lists every named root
(`${root}` in a config, `PSR-04`-`PSR-07`) bound on this machine, with the
path it currently resolves to and which layer supplied the binding --
`SMFTOOLS_ROOT_<NAME>`, the user roots file, or a `roots.toml` walked up from
`--config-dir`. A root bound to more than one candidate location (`PSR-16`,
`analyses = ["path/a", "path/b"]`) shows every candidate underneath its name,
marking whichever one it currently resolves to. See the directory
organization tutorial's Portability section for the full resolution story;
this command only ever reads bindings, it does not set them -- there is no
`data roots set` yet.

## Volume identity

`smftools data` is a third top-level group, below any single experiment and
across all projects, for machine- and volume-scoped storage operations (portable
storage roots -- `PSR`). `smftools data init-volume MOUNT --label LABEL [--kind
{working,archive,backup}]` stamps a drive with a permanent `.smftools-volume.json`
identity file at its mount root. Because the stamp travels with the drive, it
identifies the volume on any machine it is later attached to, independent of
mount point or OS-reported volume name.

The stamp is written once and never rewritten: re-running `init-volume` on an
already-stamped mount leaves it untouched and reports its existing identity
(warning if the requested `--label`/`--kind` differ from what is already
recorded), so a drive keeps its `volume_id` even if it is relabeled at the OS
level or reattached under a different mount point. `label` and `kind` are
user-facing metadata only; nothing derives `volume_id` from them.

`smftools data volumes [--config-dir DIR] [--json]` lists every stamped volume
currently attached to this machine. Discovery scans the platform's mount roots
(`/Volumes` on macOS; `/mnt`, `/media/<user>`, `/run/media/<user>` on Linux)
plus any `[volumes]` `extra_search_paths` configured in `roots.toml` (or the
`SMFTOOLS_VOLUME_SEARCH_PATHS` environment variable, which takes priority and
replaces rather than adds to the file-configured paths) -- the mechanism a
network mount needs, since it usually lives outside those conventions. This
reports only what is attached right now; a stamped volume that is not
reachable is invisible to `data volumes` specifically, though `data locate`
below can still name it by `volume_id` through the replica catalog.

## Replica catalog: scan, locate, verify

A **dataset** is identified by its input-manifest digest -- the same
relocation-invariant digest `smftools.informatics.input_manifest` already
computes -- and the **replica catalog** (a plain JSON file next to
`roots.toml`) records which stamped volumes hold a copy of which dataset.

`smftools data scan [MOUNT...] [--catalog-path FILE]` walks one or more
stamped volumes for published input manifests
(`raw_outputs/input_manifest/resolved_input_manifest.json`) and registers one
replica per run root found: `(volume_id, run root's path relative to the
volume's own mount point)`. With no `MOUNT` given, it scans every volume
`data volumes` currently finds attached. Each named `MOUNT` must already be
stamped (`data init-volume` first). The same walk also registers each run
root it finds into the *analysis-location* catalog (below), keyed by the
run's own `experiment_uid` rather than its dataset digest.

`smftools data locate TARGET [--catalog-path FILE]` looks up every catalogued
replica of `TARGET`'s dataset and reports which are currently attached --
`TARGET` is a run root directory, a `resolved_input_manifest.json` path, or a
bare SHA-256 digest. This answers while every replica's volume is unplugged,
which is the point of a catalog.

`smftools data verify TARGET [--volume VOLUME_ID] [--catalog-path FILE]`
re-checksums a dataset's declared raw sources against every currently
attached replica (or just one, with `--volume`), bypassing the checksum cache
`resolve_input_manifest` uses for cheap re-ingestion -- a file corrupted
without its mtime changing is exactly the failure mode checksum verification
exists to catch, so the cache would defeat the point. A declared source that
is not currently reachable is reported as `unreachable`, not a failure --
archived raw input being offline is the expected case. Exits non-zero if any
reachable source's checksum no longer matches its manifest.

## Analysis-location status across copies

Two copies of a run's *analysis* tree are not interchangeable the way raw
replicas are -- each may hold different generations, since analysis can
happen independently at each location -- so a second, separate catalog
tracks where copies of a run's analysis tree are, keyed by the run's durable
`experiment_uid` (assigned once at raw ingestion, unlike a path or the
human-chosen `experiment_id` label).

`smftools data status [TARGET...] [--catalog-path FILE]` reports, per run:
every catalogued analysis location and whether it's currently attached;
pairwise locality between attached locations -- `identical`, `ahead`/
`behind` (one location's generation set is a superset of the other's, per
stage), `diverged` (each holds a generation the other lacks), or
`pointer_conflict` (same generations, different `current.json`) -- computed
fresh from each location's published `generations/` set, never modification
time; and, when at least one location is attached and reachable, its raw
dataset's digest and catalogued replicas. Each `TARGET` is a run root
directory or a bare `experiment_uid`; omitted, every run `data scan` has
found is reported.

Divergence and pointer conflicts are reported, never resolved -- there is no
flag that picks a side by timestamp or otherwise.

`smftools data sync TARGET [--from VOLUME_ID --to VOLUME_ID] [--dry-run]`
resolves *ahead*/*behind*, per stage, by copying whichever generations are
missing from the location that lacks them -- safe and resumable, since
generations are immutable and content-addressed, so a copy that is
interrupted partway is simply retried whole on the next run rather than
risking a corrupt partial directory. `current.json` is never moved by sync;
advancing a pointer is a separate, explicit act. With no `--from`/`--to`,
exactly two of `TARGET`'s catalogued locations must currently be attached, or
the request is refused rather than guessing which pair was meant. A
`diverged` or `pointer_conflict` stage copies nothing and is reported, the
same as in `data status`; `data sync` exits non-zero when any stage was left
unresolved this way, so a script can tell "fully synced" from "some stages
need a human".

**A known gap**: the catalogs above are populated by `data scan` walking
attached volumes -- a reconciliation mechanism, explicitly by design, for
whatever smftools did not itself just publish. Hooking every stage's publish
path to update the catalog *in band*, so a freshly-published generation
shows up in `data status` without a scan, has not been built. Run `data
scan` after publishing new work to keep `data status` current in the
meantime.

Whether or not a run has ever been scanned, offline/missing classification
itself becomes exact when both a published input manifest and a populated
catalog are available: `ExperimentConfig.from_var_dict` then distinguishes a
dataset with no attached replica anywhere (confidently `offline`, even for a
path under no recognized mount convention) from one whose volume simply
reattached under a different mount point or name (`present`, transparently,
at its new location) -- falling back to the structural guess exactly as
before whenever either input is missing. See `dev/plans/completed/
portable_storage_roots_implementation_plan.md`'s `PSR-12` for the detail.

## Localizing a config's small inputs

`smftools data localize CONFIG_PATH [--apply] [--out PATH] [--json]` copies
`fasta`, the BED region files (`alignment_regions_bed`, `analysis_regions_bed`,
`plot_regions_bed`), the sample sheet, and any barcode/UMI YAML into the
config's own `output_directory` -- never the raw input itself, which is the
large data this whole plan exists to leave archived. This is the cheapest way
to make a single experiment's `analyses/` tree self-contained: no named root,
volume stamp, or replica catalog required to read it on another machine.

Without `--apply`, this is a dry run: it reports which fields would be copied
and their total size, and touches nothing. `--apply` copies each file into
`output_directory/localized_inputs/` and writes a **new** config with those
fields repointed at the copies -- the original config is never modified.
Re-running `--apply` is safe: a destination that already holds byte-identical
content is left alone, and only a destination with genuinely different
content raises rather than being silently overwritten.

## Scaffolding a new lab tree

`smftools data init LAB_ROOT [--stamp-volume [--label LABEL] [--kind KIND]]`
creates `data/` and `analyses/{runs,projects}/` under `LAB_ROOT`, mirroring
`project init` one level up -- see
[Organizing data, experiments, and projects](tutorials/directory_organization.md)
for what belongs in each. Idempotent: re-running it only fills in whatever is
still missing, and it never touches data already collected under `data/`.

`--stamp-volume` also gives `LAB_ROOT` a permanent volume identity
(`data init-volume`, `PSR-08`). This only makes `LAB_ROOT` discoverable by
`data volumes`/`data scan` elsewhere on the machine when `LAB_ROOT` *is* the
removable volume's own mount point (e.g. `/Volumes/lab-drive`) rather than a
subdirectory of a larger one -- discovery only looks one level below a
platform mount root.

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

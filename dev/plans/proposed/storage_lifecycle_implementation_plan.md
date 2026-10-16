# Storage lifecycle: model store, generation pruning, one usage report

Status: proposed. Items `SLC-01`--`SLC-12`. Extends `MLR-16` (model
retention, merged) and replaces the deletion blocker of `EGL-03b`
(`completed/generation_lifecycle_and_naming_implementation_plan.md`); folds in
`EGL-03c` (`proposed/generation_prune_scope_implementation_plan.md`).

## Why

On one analysis drive the disk filled (under 0.2 % free of 3.6 TiB) and a
stage regeneration could not start. Measured:

- **Runs (1.0 TB, 21 experiments).** Every stage invocation publishes a new
  generation and keeps the old ones. A 1.3M-read deaminase run holds 183 GB,
  of which ~18 GB are the current generations and ~115 GB superseded ones
  (raw 4 x 12 GB, preprocess 4 x 11 GB, spatial 9 GB, HMM 14 GB).
- **One project's ML store (184 GB).** 6,451 model bundles, 187 GB of them
  random forests (2,439 x ~77 MB). `MLR-16` retention had already retired
  1,251 superseded models (70 GB); what is left belongs to runs nothing
  supersedes. Only 860 of the 6,451 models were ever read by a later run
  (explain / apply); the rest were written, scored and never opened again.
- **Unmanaged output trees (~300 GB).** Output directories from earlier
  smftools versions and test reruns sit beside the registered experiment
  roots (one run: 129 GB of them next to a 112 GB current tree). No
  generation, manifest or registry tracks them, so no tool reports or
  prunes them.
- **Payloads are uncompressed.** skops writes a zip with no compression by
  default and smftools calls it that way: the largest forest, 409 MB as
  stored, is 55 MB deflated (7.4x, lossless).
- **The prune planner's notion of "safe" is wrong.** Its protections are
  current, pinned, newest-N and recent. In the run above every current
  downstream spine (preprocess, spatial, HMM, latent) reads its data from a
  raw and a preprocess generation that are *not* current: the current
  preprocess generation is an annotation refresh (1.6 GB) of an older one
  (11 GB), and the current raw generation is a refresh whose rows live in an
  older raw generation. A deletable planner would offer exactly those two,
  and deleting them would break every stage at once.

Nothing deletes today except `MLR-16`; nothing compresses; nothing reports
the whole picture. This plan adds, in order of risk: compression, an
honest inventory, reference-aware protection, then deletion / archiving.

## Principles

- **Records stay, payloads go.** Manifests, run records (metrics,
  predictions, curves), generation manifests and a removal record are kept
  for everything; what is reclaimed is bulk payload (model files, stage
  stores). A later load of a removed payload fails with the reason, never
  "not found".
- **Reachability, not age, decides what is safe.** An artifact is needed if
  something live reads it, transitively. Age and count policies only choose
  among the unreachable.
- **Planning writes nothing; applying is explicit.** Every destructive step
  has a dry-run plan, then `--apply` (with archive as the default way to
  apply), re-checked under the owning lock immediately before it acts.
- **Archive before delete.** Moving a payload to another volume frees the
  space and loses nothing; deletion is a separate, confirmed step.

## Items

### A. ML model store

| id | item | status |
|---|---|---|
| `SLC-01` | compressed payloads for new models: skops `compression=ZIP_DEFLATED` (level 6) in `sklearn_artifacts`; torch checkpoints likewise where the format allows; setting `ml_model_compression: deflate \| none` (default deflate). New models only -- identities of existing ones untouched | proposed |
| `SLC-02` | compact existing payloads: `project ml-storage compact [--apply]` rewrites stored skops zips compressed. Model identity hashes the payload reference (sha256, size), so the manifest gains a `storage` block (`encoding`, `stored_sha256`, `stored_size`) while the identity keeps the original reference; compaction verifies every zip member's uncompressed bytes against the original before replacing it, atomically, under the publication lock; validation checks `stored_sha256` when present | proposed |
| `SLC-03` | persistence policy per job: `persist_models: all \| needed \| none` (default `all`, as now). `needed` keeps payloads only for models a later job of the same plan reads (explain / apply / transfer sources) or an alias names; `none` keeps the model manifest with `payload: not_persisted`. A plan whose later job reads a model that would not be persisted fails validation before training | proposed |
| `SLC-04` | payload retirement for kept runs (extends `MLR-16`): `project ml-retention --unread [--older-than]` retires payloads of models no run has read, no alias names and no kept model descends from, even when their run is not superseded; the run record and model manifest stay (`retired_models.json` gains `reason: unread`). Optional refit check: a retired model records dataset snapshot, split and seed, so it can be refit | proposed |
| `SLC-05` | archive mode for retention: `--archive DIR` moves payloads instead of deleting them; `retired_models.json` records the archive location; `project ml-storage restore MODEL_ID` brings one back | proposed |

### B. Stage generations

| id | item | status |
|---|---|---|
| `SLC-06` | generation reference graph (read-only, `informatics/generation_references.py`): pointer edges -- any `<stage dir>/generations/<id>` in a generation's spine `uns` or its small JSON files (depth <= 2, <= 4 MB; stores never opened) -- and lineage edges -- manifest keys ending in `generation_id` naming another generation, except bookkeeping (`current_generation_id`). `experiment generations ROOT refs [--json]`. On the measured run it finds every hand-traced edge (23) and protects exactly the non-current raw and preprocess generations the live spines read | implemented |
| `SLC-07` | planner protection: `keep_referenced` for everything reachable from current, pinned and `--protect-from` generations (a project directory reads its `registry.json`), with the read that protects it; `keep_references_unknown` for every non-current generation when any generation's references cannot be read; the byte-reproducibility blocker is replaced by `prune_candidate` (`deletion_allowed` true, `reclaimable_bytes` = candidates; plan schema 2, still `dry_run` / `deletion_supported: false` until `SLC-08`) | implemented |
| `SLC-08` | apply: `experiment generations ROOT prune ... --archive DIR` (default way to apply) or `--delete --confirm ID[,ID]` (ids must match the plan). Holds the stage lock, refuses while a stage of that experiment runs, re-plans before acting; leaves a stub (`generation_manifest.json` plus `pruned.json`: when, why, archive location, size) so inventory shows the generation as `archived` / `deleted`; `generations ROOT restore KIND ID` reverses an archive | proposed |
| `SLC-09` | project and directory scope (`EGL-03c`): `project generations DIR prune` and `data prune DIR`, per-experiment policy evaluation, per-experiment breakdown in the report; same apply semantics | proposed |
| `SLC-10` | space check before a stage starts: estimate the new generation's size from the current one (or the raw inputs for a first run) and refuse to start with less free space than that times a margin (`min_free_space_factor`, default 1.5), naming the reclaimable bytes the planner reports. A run that fills the disk midway leaves partial stores and wasted hours | proposed |

### C. One report

| id | item | status |
|---|---|---|
| `SLC-11` | `smftools data usage DIR [--json]`: per experiment and stage -- current, referenced, superseded, pinned bytes; per ML workspace -- models by architecture and job tags, read vs never read, compressed vs not, retired; project analysis caches (`project analyses list`) current vs stale; output trees under a run directory that no registered experiment root, project registry or generation pointer reaches (reported with size and whether they carry an `experiment_manifest.json`, never deleted by smftools -- other projects may read them); and for each bucket the command that reclaims it | proposed |
| `SLC-12` | docs: a "managing disk space" page (what grows, retention order: compact, retire unread, prune superseded generations, archive), config reference for the new settings, `cli/AGENTS.md` command map | proposed |

## Order

1. `SLC-01`, `SLC-11` (no risk; compression stops new growth, the report
   shows where the bytes are).
2. `SLC-06`, `SLC-07` (read-only; makes the generation plan trustworthy).
3. `SLC-02` (rewrites payloads, verified member-by-member).
4. `SLC-05`, `SLC-08` archive paths; then `SLC-03`, `SLC-04`, `SLC-10`.
5. `SLC-09`, `SLC-12`.

Expected on the measured drive: compaction ~185 -> ~30 GB of models;
superseded generations ~90 GB on the run above alone; a space check would
have refused the regeneration that could not finish.

## Tests

- Compression round-trip: compressed and uncompressed payloads load to
  identical predictions; `SLC-02` refuses to replace a payload whose members
  differ; identities of compacted models unchanged; validation catches a
  corrupted compacted file.
- Persistence policy: a plan whose explain job reads an unpersisted model
  fails validation; `needed` keeps exactly the read set.
- Reference graph: a fixture where the current preprocess generation is an
  annotation refresh and the current downstream spines read a non-current
  raw generation -- every one of those is `keep_referenced`; a registry
  pointer protects its target; cycles and missing targets are reported, not
  followed.
- Apply: archive then restore yields a byte-identical generation; delete
  refuses ids not in the plan, a running stage, a generation that became
  referenced after planning; stubs show in inventory.
- Space check: refusal message names required, free and reclaimable bytes.

## Open questions

- `SLC-02` identity: keep the original payload reference as identity and
  verify the stored file separately (proposed), or move identity to a
  canonical member-level digest for new models so storage encoding never
  enters identity? The latter is cleaner for future encodings but changes
  `ModelManifest` identity for new models.
- Should `persist_models` default to `needed` for grid sweeps
  (`MLR` jobs with many model keys), keeping `all` for single trainings?
- Do project-owned generations (embedding caches) carry current / pin state
  the planner can use (the `EGL-03c` open question)?

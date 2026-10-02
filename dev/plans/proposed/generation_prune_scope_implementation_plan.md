# Generation prune scope expansion (`EGL-03c`)

**Status:** proposed. No implementation branch yet.

**Predecessor:** `completed/generation_lifecycle_and_naming_implementation_plan.md`
(`EGL-03` -- retention, pinning, and pruning). `EGL-03a` shipped a read-only,
dry-run-only prune planner (`smftools experiment generations <output_root>
prune`) scoped to a single experiment's `output_root`. `EGL-03b` (actual
deletion) remains an explicit, unresolved, deliberately deferred blocker --
that document's own words: pruning stays blocked "until byte-level
reproducibility has an authoritative representation in generation
provenance," and the standing-blocker note calls it unchanged and not
urgent. **This plan does not touch `EGL-03b` in any way.** It stays a pure
read-only planning feature, just at wider scope. `deletion_allowed` stays
hardcoded `False` at every decision, exactly as `EGL-03a` shipped it.

## Problem

`EGL-03a`'s planner (`plan_experiment_generation_prune` in
`informatics/generation_pruning.py`) only accepts one experiment's
`output_root`. Answering "how much space could pruning reclaim across a
project" or "across everything on this drive" today means running
`experiment generations <output_root> prune` once per experiment by hand and
adding the numbers up. `project generations` (listing, `EGL-02`) already
solved exactly this fan-out problem for inventory; pruning never got the
equivalent, and there is also no way to plan pruning over an arbitrary
directory of run roots that isn't a registered project at all.

## Design

**No new policy semantics.** `keep-last`/`older-than`/pin/current protection
stay exactly what `EGL-03a` defined, and stay evaluated *within one
experiment* -- `--keep-last 3` means the newest 3 generations of a kind *in
that experiment*, never a global ranking across a project or directory.
Ranking globally would let one active experiment's recent generations crowd
out another experiment's only generations from ever being "kept," which is
not what a project- or directory-wide prune request means.

**Reuse existing discovery and fan-out precedent; do not reinvent either.**

- *Project scope* reuses the exact fan-out `project_generations()` already
  uses in `cli/generations.py`: resolve registered experiments via
  `project_list()`, run the (unchanged) per-experiment planner against each
  reachable one, and skip an experiment whose registered path is unreachable
  (unmounted volume, moved tree) rather than raising -- a partial plan is the
  useful answer, the same precedent `project_generations()` already
  established, and the gap stays visible by comparing against `project
  list`.
- *Directory scope* reuses `data/volume_scan.py::_iter_run_roots(mount)`,
  which already walks an arbitrary directory tree for every
  `experiment_manifest.json`-marked run root (built for `data scan`, generic
  enough to reuse as-is) -- promoted from a private helper to a small shared
  one rather than duplicated.
- *CLI group conversion* reuses `_GenerationGroup` (`cli_entry.py`), the
  `click.Group` subclass already written specifically to let a
  `ROOT --json`/`ROOT --size` call keep working after a plain command grows
  subcommands -- built for exactly this migration when `experiment
  generations` itself gained `pin`/`unpin`/`prune`. `project generations`
  reuses the same class rather than re-solving flag-ordering.

**Aggregation, not a new plan shape.** `plan_experiment_generation_prune`
gets split into a pure decision function over an already-fetched
`list[GenerationRecord]` and a thin single-experiment wrapper that calls it
-- today's public function, signature and behavior unchanged. Two new
functions call the same pure decision function per experiment and wrap the
results:

```text
plan_project_generation_prune(project_dir, *, keep_last, older_than, stages) -> AggregatePrunePlan
plan_directory_generation_prune(directory, *, keep_last, older_than, stages) -> AggregatePrunePlan
```

`AggregatePrunePlan` holds one `PrunePlan` per experiment plus summed
`candidate_bytes`/`reclaimable_bytes` (the latter staying 0 everywhere, same
as today -- `EGL-03b` is untouched). Rendering (table and `--json`) shows the
per-experiment breakdown so a reader can see which run is holding the bytes,
not just a grand total.

**CLI surface:**

- `project generations <project_dir> prune [--stage] [--keep-last]
  [--older-than] [--json]` -- `project generations` becomes a group (reusing
  `_GenerationGroup`) so `pin`/`unpin`/`prune` could all eventually live
  there the same way `experiment generations` does; this plan only adds
  `prune`.
- `data prune <directory> [--stage] [--keep-last] [--older-than] [--json]`
  -- new, parallel to `data scan`, for an arbitrary directory of run roots
  that is not a registered project.

**What stays out of scope.** Actual deletion (`EGL-03b`) is not this plan's
job. Project-owned generations (the embeddings container
`list_project_generations` also inventories) get evaluated read-only and
non-destructively too, if and only if they carry their own pin/current
state comparable to an experiment's stage containers -- see Open questions.

## Work items

| item | status | evidence |
|---|---|---|
| `EGL-03c-1` split `plan_experiment_generation_prune` into a reusable decision function + thin wrapper, no behavior change | proposed | -- |
| `EGL-03c-2` `plan_project_generation_prune`, fanning across registered experiments (and project-owned generations, if they carry retention/current state) | proposed | -- |
| `EGL-03c-3` `plan_directory_generation_prune`, reusing `_iter_run_roots` | proposed | -- |
| `EGL-03c-4` CLI: `project generations` becomes a group (`_GenerationGroup`); add `project generations prune` | proposed | -- |
| `EGL-03c-5` CLI: `data prune <directory>` | proposed | -- |
| `EGL-03c-6` aggregate table/JSON rendering with per-experiment breakdown | proposed | -- |
| `EGL-03c-7` tests: multi-experiment project fixture, directory-of-runs fixture, skip-unreachable-experiment behavior, per-experiment `keep-last` isolation, `deletion_allowed == False` regression guard | proposed | -- |
| `EGL-03c-8` docs: `cli/AGENTS.md` command map, changelog/migration note for the new CLI surface | proposed | -- |

## Open questions

- Do project-owned generations (`list_project_generations`'s embeddings
  container) have their own `current.json`/`retention.json`, the same as an
  experiment's stage containers? If not, `plan_project_generation_prune`
  needs an honest, defined behavior for them -- e.g. reported as
  `not_applicable` -- rather than silently omitting them or guessing a
  policy that does not exist for that container. Needs an answer before
  `EGL-03c-2` is implemented, not during.

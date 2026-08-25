# Working in `dev/plans/`

Design records for smftools. Tracked in git and public, except `logs/`.

## Where a document goes

There are three **kinds** of document, and only one of them has a status.

| directory | kind | tracked |
|---|---|---|
| `audits/` | investigation of the code as it is | yes |
| `proposed/` `in-progress/` `completed/` | plans, filed by status | yes |
| `logs/` | append-only records that never "complete" | **no** |

**An audit never completes.** It is a snapshot of the code at a moment: it can
go *stale*, but it is never "done". Filing one under `completed/` claims it is
settled when what finished was the plan it motivated -- so audits live in their
own directory and carry a staleness marker instead of a status:

```markdown
**Audited against `a5dc558`** (2026-08-20).
```

A plan moves between `proposed/`, `in-progress/` and `completed/` as its status
changes. `logs/` is gitignored because it is where raw measurements from
unpublished experiments land first.

The usual unit is a **pair**: `audits/<program>_audit.md` investigates, and
`<status>/<program>_implementation_plan.md` tracks the work. They live in
different directories and link to each other; the plan cites the audit that
motivated it, and the audit names the plan it produced.

## Two hard rules

**No sequencing-run names.** Cite a run's scale and modality -- "a 1.3M-read
deaminase run" -- never a `<YYMMDD>_<description>` run identifier. Runs are
unpublished research data; a design document needs the shape of the input, not
its identity. A test enforces this too, and it applies to this file: writing a
real run name here as the worked example is how the first version of the guard
published one.

**No absolute paths.** Repo-relative only (`dev/plans/...`, `src/smftools/...`),
or a placeholder (`<repo>`, `<outputs>`). A test enforces this; it exists because
four of them survived in a document for months.

**Cite findings, not datasets.** Write "measured on a 1.3M-read deaminase run",
never a run identifier. Experiment names are unpublished research data, and a
reader of a design document needs the *scale and modality*, not the identity.
Where a document genuinely needs to locate inputs, that mapping is operator
state: keep it in the analyses repository and point at it.

## Identifiers

Work items get a stable prefixed id -- `EGL-04`, `NKG-03`, `F33`. **Cite by id,
not by filename**: ids survive file moves and renames, links do not. Use
relative links for reading, ids for precision.

Prefixes are per program (`EGL` = generation lifecycle, `NKG` = regeneration,
`F` = findings). Pick a new one for a new program; never renumber.

## What a finding must contain

The template below is the one that has repeatedly prevented rework. The third
part is the one usually omitted and the most valuable.

```markdown
### F41 — The run gate ignored `algorithm_version`

**Found 2026-08-24.** `smftools experiment plan` reported `stale_algorithm` and
the run skipped the stage anyway: `partitioned_stage_is_complete` compared config
hash, inputs and artifacts but not the version.

**Why it hid.** The previous rebuild coincided with a config change, which moved
the config hash and forced the rerun for an unrelated reason. A mechanism that
works only when something else independently forces the same outcome is
indistinguishable from one that works.

**Fixed 2026-08-24** (`ae1d206`). Folded the declared version into the gate.

**A wrong attempt, recorded so it is not retried.** <what was tried, why it
looked right, and what disproved it.>
```

Rules that fall out of it:

- **Date what was found and what was fixed**, and cite the commit or PR.
- **Record wrong attempts.** They are the highest-value content: they stop the
  next person re-deriving a dead end.
- **State what a claim rests on.** "Verified by a side-by-side run" and "rests on
  an associativity argument plus unit tests" are different claims; say which.
- **Quote measurements, not impressions.** "65.4 GiB before any task dispatched"
  beats "used a lot of memory".

## Size

One document should cover one thing that can finish. When part of it will never
finish, that part is a log and belongs in `logs/`. A document twice the length of
its neighbours is usually several programs fused together.

## Status tables

Lead a plan with a table of `item | status | evidence`, where evidence is a PR
number or commit sha. Prose drifts; a table with shas can be checked.

"""Human-facing starter files for a new project directory.

``smftools project init`` writes :data:`registry.json`/``sets/`` (the machine-managed
state -- see :mod:`.registry`) plus a handful of documentation starting points: a
README, ``AGENTS.md``/``CLAUDE.md`` working-context files for coding agents, a
``PLAN.md`` for tracking the current objective, and a ``project.yaml`` manifest for
human-curated run/reference notes alongside the registry -- plus two working
directories, ``project_scripts/`` (project-specific drivers/constants, importable
as a package) and ``project_outputs/`` (materialized/derived outputs). None of
these are ever read back by smftools itself -- they exist so a project directory
is immediately useful to work in, whether by the user or a coding agent, instead
of starting from an empty folder. Never overwrites a file or directory that
already exists, so re-running ``project init`` on an existing project only fills
in whatever is still missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _readme_template(name: str) -> str:
    return f"""# {name}

<!-- One or two sentences: what this project is about (locus/target, modality, question). -->

## Overview

- **Modality**: <!-- direct / conversion / deaminase -->
- **Reference(s)**: see `project.yaml`
- **Registry**: `registry.json` is managed by `smftools project add`/`list`/`materialize` --
  do not hand-edit it.

## Directory structure

```text
{name}/
├── project.yaml         # Human-curated run/reference manifest (not read by smftools)
├── registry.json        # Machine-managed experiment pointers (smftools project init/add)
├── sets/                # Named experiment subsets (smftools project add-set)
├── runs/                # Symlinks -> analyses/runs/<run_name>/<date>_outputs (no data here)
├── project_scripts/     # Project-specific drivers, constants, orchestration
├── project_outputs/     # Materialized/derived outputs (project materialize -o, figures, ...)
├── README.md            # This file
├── AGENTS.md            # Working context for coding agents
├── CLAUDE.md            # Points Claude Code at AGENTS.md
└── PLAN.md              # Current objective / status / next steps
```

## Registering experiments

```shell
smftools project add . /path/to/analyses/runs/<run_name>/<date>_outputs --id <run_name>
```

A legacy monolithic `.h5ad`/`.h5ad.gz` (pre-partitioned-store run) can be registered
the same way, with `--stage` naming which pipeline stage it represents.

## Querying across the project

```shell
smftools project list .
smftools project materialize . <canonical_reference> -o project_outputs/<canonical_reference>.h5ad.gz
```

See smftools's directory organization guide for the full experiment -> project
workflow, cross-stage visibility, and portability conventions this layout follows.
"""


def _agents_template(name: str) -> str:
    return f"""# AGENTS.md

This file provides shared working context for coding agents in this project directory.

## Project

<!-- What locus/target, modality, and question(s) does {name} cover? -->

`smftools` is the installed Python package behind this project's pipeline (BAM ->
AnnData) and analysis layer. Generic, reusable analysis logic belongs in
`smftools.analysis`; project-specific drivers, constants, and orchestration belong
in `project_scripts/`, which imports from it.

## Working layout

```text
{name}/
├── project.yaml
├── registry.json
├── sets/
├── runs/                   # symlinks to analyses/runs/<run_name>/<date>_outputs
├── project_scripts/        # project drivers, constants, orchestration
├── project_outputs/        # materialized/derived outputs, figures
├── README.md
├── CLAUDE.md
├── AGENTS.md
└── PLAN.md
```

## Code boundaries

```text
smftools.cli               pipeline: BAM -> AnnData
smftools.analysis          generic compute, filters, plotting
{name}/project_scripts/    project drivers, constants, orchestration
```

Rules:
- Put reusable logic in `smftools.analysis`, not `project_scripts/`.
- Keep `project_scripts/` project-specific: paths, sample groupings, analysis entrypoints.
- Avoid hardcoded sample names or run-specific paths in reusable functions.
- Write materialized/derived outputs (`project materialize`, figures, caches) under
  `project_outputs/`, not scattered elsewhere.

## Data model

Registered experiments are queried via `smftools project materialize` (see
README.md); each experiment's most-derived pipeline stage (HMM > spatial >
preprocess > raw) is used by default, or pin one with `--stage`.

<!-- Once conventions settle, record here: primary obs columns, key layers, and
     what the obs/var axes mean for this project's modality. -->

## Sample metadata

<!-- Where the per-sample/per-barcode metadata sheet lives (e.g. samples.csv) and
     its key columns, once one exists. -->

## Reference genome(s)

<!-- See project.yaml for the authoritative list; summarize here once it's populated. -->

## Run commands

```bash
smftools project list .
smftools project materialize . <canonical_reference> -o project_outputs/<canonical_reference>.h5ad.gz
```

## Working norms

- Read `CLAUDE.md` for pointers to fuller context.
- Read `PLAN.md` before starting substantial work.
- Update `PLAN.md` when the current objective, blockers, or next step changes.
- Preserve existing user changes; do not revert unrelated work.
"""


def _claude_template() -> str:
    return """# Claude Code Agent Instructions

You are the implementation agent defined in this project's AGENTS.md.
"""


def _plan_template() -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"""# PLAN.md

## Current Objective

<!-- What are we trying to accomplish right now? -->

## Current Status

- State: not started
- Owner:
- Last updated: {today}

## Context To Reuse

- Read `AGENTS.md` for project-level working context.
- Read `README.md` for the scientific/project overview.

## Open Questions

<!-- Anything blocking a decision. -->

## Next Steps

1.

## Validation

<!-- Append one line per verified fact/result as work proceeds. Keep entries
     factual and dated. -->

## Notes

- Keep this file short and current.
"""


def _project_yaml_template(name: str) -> str:
    return f"""name: {name}
description: "TODO: one paragraph on what this project covers (locus/target, modality, question)"

# One entry per reference FASTA used across registered runs.
references: []
# Example:
# references:
#   - id: base
#     fasta: reference.fasta        # path relative to this directory
#     description: what this reference covers
#     sequences:
#       - name: chr1                # record name as it appears in the FASTA
#         length_bp: 0
#         description: what this sequence is

# One entry per registered experiment. registry.json (managed by `smftools
# project add`) is what `smftools project` commands actually read; this file is
# a human-curated companion for narrative context alongside it.
runs: []
# Example:
# runs:
#   - name: 250101_my_run           # matches runs/<run_name>
#     date: 2025-01-01
#     reference: base               # references[].id above
#     description: what this run is
"""


_SCAFFOLD_TEMPLATES = {
    "README.md": _readme_template,
    "AGENTS.md": _agents_template,
    "PLAN.md": lambda name: _plan_template(),
    "project.yaml": _project_yaml_template,
    "CLAUDE.md": lambda name: _claude_template(),
}


# Project-local working directories: drivers/constants/orchestration, and
# materialized/derived outputs -- kept separate from `runs/` (pointers only, no
# data) so it's clear at a glance what's safe to delete/regenerate vs. what's
# project-specific code worth version-controlling.
_SCAFFOLD_DIRS = ("project_scripts", "project_outputs")


def scaffold_project(project_dir: str | Path, *, name: str | None = None) -> list[Path]:
    """Write starter README/AGENTS/CLAUDE/PLAN/project.yaml files and working directories.

    Skips anything that already exists (files or directories), so re-running this
    on an existing project only fills in whatever is still missing. Returns the
    paths actually created, for callers to report what's new.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    project_name = name or project_dir.resolve().name

    written: list[Path] = []
    for dirname in _SCAFFOLD_DIRS:
        dir_path = project_dir / dirname
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            written.append(dir_path)

    init_path = project_dir / "project_scripts" / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""Project-specific analysis drivers, constants, and orchestration."""\n'
        )
        written.append(init_path)

    for filename, template in _SCAFFOLD_TEMPLATES.items():
        path = project_dir / filename
        if path.exists():
            continue
        path.write_text(template(project_name))
        written.append(path)
    return written

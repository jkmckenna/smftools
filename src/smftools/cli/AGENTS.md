# AGENTS.md — src/smftools/cli

CLI implementation for the `smftools` command. See root AGENTS.md first for repo-wide policy;
this file covers conventions specific to this subpackage.

## Structure

`src/smftools/cli_entry.py` defines the Click groups and thin command wrappers.
`src/smftools/cli/*.py` holds the actual per-stage implementation, one module per stage/concern.

Every pipeline-stage command follows the same split:

- `<stage>_adata(config_path)` — the Click-facing wrapper. Resolves config, decides whether the
  stage needs to run at all (stage-skip / already-done checks), and if so calls
  `<stage>_adata_core(...)`.
- `<stage>_adata_core(...)` — the real logic. This is almost always the function you actually
  want to read or edit; the outer wrapper is boilerplate.

Exception: `raw_adata()` (in `raw_adata.py`) does not have its own `_core` — it delegates to
`load_adata_core(cfg, paths, config_path=config_path, raw_only=True)` in `load_adata.py`, since
raw ingestion and dense-cache loading share the same underlying function.

## Command map

`smftools --help` is authoritative; this is a summary. Two top-level groups:

### `smftools experiment <config_path>` — pipeline stages for a single experiment

| Command | Module | Core function | Purpose |
|---|---|---|---|
| `raw` | `raw_adata.py` | `load_adata_core(..., raw_only=True)` | Prepare BAM artifacts and write the ragged raw store. |
| `load` | `load_adata.py` | `load_adata_core` | Optionally pre-build the dense zarr cache from raw artifacts. |
| `preprocess` | `preprocess_adata.py` | `preprocess_adata_core` | QC, filtering, read-level preprocessing. |
| `variant` | `variant_adata.py` | `variant_adata_core` | Sequence variation analyses. |
| `chimeric` | `chimeric_adata.py` | `chimeric_adata_core` | Detect putative PCR chimeras. |
| `spatial` | `spatial_adata.py` | `spatial_adata_core` | Spatial signal analysis. |
| `hmm` | `hmm_adata.py` | `hmm_adata_core` | HMM feature annotation and plotting. |
| `latent` | `latent_adata.py` | `latent_adata_core` | Latent representations (PCA/UMAP/NMF/CP). |
| `full` | `recipes.py` | — | Composed workflow: raw, preprocess, spatial, hmm. |
| `batch` | `cli_entry.py` | — | Run one stage across many experiments from a CSV/TSV/TXT. |
| `concatenate` | `cli_entry.py` | — | Merge multiple `.h5ad` files into one. |
| `export-fastq` | `export_fastq.py` | `export_fastq_for_experiment` | FASTQ export of QC-passed reads, per experiment. |
| `plot-current` | `plot_current.py` | — | Plot nanopore current traces for specified reads. |

### `smftools project <project_dir>` — registering and querying across experiments

| Command | Purpose |
|---|---|
| `init` | Initialize a project directory + registry. |
| `add` / `remove` | Register or deactivate an experiment in the project. |
| `list` | List registered experiments and harmonized references. |
| `materialize` | Pool a canonical reference across matching experiments into one AnnData. |
| `sample-store-list` | List cataloged per-sample-store partitions. |
| `export-fastq` | FASTQ export of QC-passed reads, across every registered experiment. |

All `project_*` commands live in `project_cmd.py`.

## Shared helpers

- `helpers.py` — `AdataPaths`/`ArtifactPaths` dataclasses (canonical per-stage file paths) and
  `resolve_adata_stage()` (stage-fallback resolution: hmm > latent > spatial > chimeric > variant
  > pp_dedup > pp > raw).
- `stage_input.py` — `StageSlice` dataclass for partition-scoped stage inputs.
- `stage_artifacts.py` — `StagePlotPaths` dataclass for per-stage output figure paths.

## When adding a new CLI command

1. Decide if it's `experiment`-scoped (one config, one experiment) or `project`-scoped (crosses
   experiments) — this determines which Click group it joins in `cli_entry.py`.
2. If it does real work beyond a thin wrapper, follow the `<name>(...)` / `<name>_core(...)`
   split so the logic is testable independent of Click.
3. Update the table above and `docs/source/cli.md`.
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

`smftools --help` is authoritative; this is a summary. Three top-level groups:

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
| `plan` | `cli_entry.py` | `pipeline/experiment_graph.plan_experiment` | Read-only compatibility/recomputation plan for a target. |
| `run` | `workflow_contract.py` | `run_experiment_workflow` | Engine-facing task-local execution with a stable result contract. |
| `validate` | `workflow_contract.py` | `validate_workflow_output` | Validate a completed or relocated workflow output without writing. |
| `export-bundle` | `export_bundle.py` | `export_bundle_for_experiment` | Sequence-only or lossless-BAM re-ingestion bundle. |

### `smftools project <project_dir>` — registering and querying across experiments

| Command | Purpose |
|---|---|
| `init` | Initialize a project directory + registry. |
| `add` / `remove` | Register or deactivate an experiment in the project. |
| `list` | List registered experiments and harmonized references. |
| `add-set` / `list-sets` / `show-set` / `remove-set` | Define and inspect named experiment sets used by `--set`. |
| `plan` | Read-only dependency plan for one target (`selection`, `materialization`, `sample-analysis`, `embedding`). |
| `run` | Execute one product task-locally; `--target` selects materialization (default), sample-analysis, or embedding. |
| `validate` | Validate a published project result against artifacts and the current source plan. |
| `materialize` | Pool a canonical reference across matching experiments into one AnnData. |
| `sample-analysis` | Per-sample periodicity across a selection, cached per partition. |
| `embedding` | Fit or extend one shared cross-experiment embedding. |
| `sample-store-list` | List cataloged per-sample-store partitions. |
| `export-fastq` | FASTQ export of QC-passed reads, across every registered experiment. |
| `export-latent` | One scoped artifact per experiment/core latent coordinate owner. |
| `export-bundle` | Project-scoped re-ingestion bundle. |

Project command logic lives in `project_cmd.py`, except `plan`/`run`/`validate`, whose
task-local execution and result/validation contracts live in `workflow_contract.py`.
`run --target` dispatches to `run_project_{materialization,sample_analysis,embedding}_workflow`
there; `selection` is a planning-only dependency of the other three and is deliberately
not executable.

### `smftools data` — machine- and volume-scoped storage operations

Below any single experiment and across all projects (portable storage roots — `PSR`).

| Command | Module | Core function | Purpose |
|---|---|---|---|
| `init-volume` | `data_cmd.py` | `data_init_volume` → `smftools.data.volume_stamp.init_volume` | Stamp a drive with a permanent `volume_id` (`.smftools-volume.json`). Idempotent: rerunning on an already-stamped mount reports the existing identity rather than rewriting it. |
| `scan [MOUNT...]` | `data_cmd.py` | `data_scan` → `smftools.data.volume_scan.scan_and_catalog` | Walk stamped volume(s) for published input manifests, registering one replica per run root into the catalog. Defaults to every currently attached volume. |
| `locate TARGET` | `data_cmd.py` | `data_locate` → `smftools.data.replica_catalog` | Every catalogued replica of TARGET's dataset and which are attached. |
| `verify TARGET [--volume]` | `data_cmd.py` | `data_verify` → `smftools.data.volume_verify.verify_replica` | Re-checksum a replica's declared raw sources directly (bypasses the ingestion checksum cache). |
| `localize CONFIG_PATH [--apply]` | `data_cmd.py` | `data_localize` → `smftools.data.localize` | Copy a config's small referenced inputs (fasta, BED files, sample sheet, barcode/UMI YAML) into its own output directory. Dry run by default; `--apply` copies files and writes a new config, never editing the original. |
| `init LAB_ROOT [--stamp-volume]` | `data_cmd.py` | `data_init` → `smftools.data.lab_init.scaffold_lab_root` | Scaffold `data/` + `analyses/{runs,projects}/` under a new lab root, mirroring `project init` one level up. `--stamp-volume` also stamps LAB_ROOT (PSR-08). |
| `status [TARGET...]` | `data_cmd.py` | `data_status` → `smftools.data.analysis_catalog`, `smftools.data.run_locality` | Where every known run's data and analyses are, attached or not, and pairwise ahead/behind/diverged/pointer_conflict locality between attached copies. |
| `sync TARGET [--from --to]` | `data_cmd.py` | `data_sync` → `smftools.data.run_sync.sync_run_locations` | Additively copy missing generations between two attached analysis locations of a run. Never moves `current.json`; diverged/pointer_conflict stages are reported, not resolved. |
| `roots list` | `data_cmd.py` | `data_roots_list` → `smftools.config.roots.known_roots` | List every named root (`${root}` in a config) bound on this machine, its resolved path, and which resolution layer supplied it. Read-only; no `data roots set` yet. |
| `archive-basecall RUN_ROOT --to ARCHIVE_ROOT` | `data_cmd.py` | `data_archive_basecall` → `smftools.data.basecall_archive.archive_basecall_generation` | Write RUN_ROOT's current basecall generation back to its POD5 archive: `basecalls/<model>@<dorado_version>/`, checksum-verified, idempotent. Reports `same_volume` (`PSR-08`) so a caller sequencing multiple runs can avoid interleaving read and write on one volume (`BCS-09`). |

Volume/data business logic lives in `smftools/data/` (parallel to `smftools/project/`),
not under `cli/`; `data_cmd.py` is the same thin Click-facing translation layer as
`project_cmd.py`.

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
4. For a new *project product* (something `project plan` can name as a target), it also
   needs an entry in `tests/acceptance/project_cli_criteria.json` and a row in the plan
   target table in `docs/source/cli.md`, which asserts that every plan target maps to one
   execution and validation lifecycle. `tests/unit/test_project_cli.py::
   test_project_plan_targets_map_to_documented_execution_paths` fails if a plan target
   gains no executor.
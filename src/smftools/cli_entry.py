from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from ._version import __version__
from .cli.recipes import full_flow
from .logging_utils import get_logger, setup_logging
from .memory_guard import enable_aggregate_memory_cap

# Single-threaded BLAS/OMP/numexpr is enforced at import time by the
# smftools package's own __init__.py (before this module -- or anything
# else in the package -- ever imports numpy/pandas). See that file's
# comment for why it has to happen there and not here.


def _configure_multiprocessing() -> None:
    import multiprocessing as mp
    import sys

    logger = get_logger(__name__)

    try:
        if sys.platform == "win32":
            mp.set_start_method("spawn")
            logger.debug("Setting multiprocessing start method to spawn")
        else:
            # try forkserver first, fallback to spawn
            try:
                mp.set_start_method("forkserver")
                logger.debug("Setting multiprocessing start method to forkserver")
            except ValueError:
                mp.set_start_method("spawn")
                logger.debug("Setting multiprocessing start method to spawn")
    except RuntimeError:
        logger.warning("Could not set multiprocessing start method")


@click.group()
@click.version_option(version=__version__, prog_name="smftools")
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional file path to write smftools logs.",
)
@click.option(
    "--log-level",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level for smftools output.",
)
def cli(log_file: Path | None, log_level: str):
    """Command-line interface for smftools."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=level, log_file=log_file)
    # Before any worker pool exists, so every process this command later forks
    # (multiprocessing children inherit their parent's cgroup) is covered.
    # No-op on non-Linux platforms; see smftools.memory_guard for why macOS
    # protection instead happens per-worker, inside the pipelines that spawn pools.
    enable_aggregate_memory_cap()
    _configure_multiprocessing()


####### Experiment-scoped pipeline stages ###########
@cli.group("experiment")
def experiment_group():
    """Run pipeline stages for a single experiment (one config path in)."""


@experiment_group.group("rebasecall")
def experiment_rebasecall_group():
    """Plan selective POD5 re-basecalling without mutating an experiment."""


@experiment_rebasecall_group.command("plan")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "request_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--json", "as_json", is_flag=True, help="Emit stable machine-readable JSON.")
def experiment_rebasecall_plan_cmd(
    config_path: Path,
    request_path: Path,
    as_json: bool,
):
    """Inspect exact parents, source signal, and selection counts without writes."""
    from .pipeline.rebasecall_plan import format_rebasecall_plan, plan_rebasecall

    try:
        plan = plan_rebasecall(config_path, request_path)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.echo(plan.to_json() if as_json else format_rebasecall_plan(plan))


@experiment_group.command("rename-id")
@click.argument("experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("new_id")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Config CSV to update (defaults to experiment_config.csv inside the experiment).",
)
@click.option(
    "--project",
    "project_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    help="Project registry to update; repeat for every project containing this experiment.",
)
def rename_experiment_id_cmd(
    experiment_dir: Path,
    new_id: str,
    config_path: Path | None,
    project_dirs: tuple[Path, ...],
):
    """Transactionally rename an experiment while preserving its durable UID."""
    from .cli.experiment_rename import rename_experiment_id

    try:
        result = rename_experiment_id(
            experiment_dir,
            new_id,
            config_path=config_path,
            project_dirs=project_dirs,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Renamed experiment {result.old_id!r} to {result.new_id!r} at "
        f"{result.experiment_dir} (UID {result.experiment_uid})."
    )
    if result.config_path is None:
        click.echo("No experiment config was found; no config file was updated.")
    if not result.project_dirs:
        click.echo("No projects were supplied; external project registries were not searched.")
    if result.query_sets_unchanged:
        click.echo(
            "Query-defined sets were left unchanged: " + ", ".join(result.query_sets_unchanged)
        )


####### Load anndata from raw data ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def raw(config_path):
    """Prepare BAM artifacts and write the ragged raw store."""
    from .cli.recipes import run_experiment_target

    run_experiment_target(config_path, "raw")


@experiment_group.command("reassemble-raw")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--no-select",
    is_flag=True,
    help="Publish the rebuilt generation without making it current.",
)
def reassemble_raw(config_path, no_select: bool):
    """Rebuild the current raw generation's obs from its existing shards.

    Re-runs only the annotation that is derivable from the shards already on
    disk -- no BAM, no alignment, no re-extraction. Publishes an immutable
    sibling generation that hardlinks the unchanged artifacts.
    """
    from .cli.helpers import load_experiment_config
    from .informatics.raw_reassembly import reassemble_raw_generation

    cfg = load_experiment_config(str(config_path))
    result = reassemble_raw_generation(cfg.output_directory, select_current=not no_select)
    click.echo(f"Published raw generation {result.get('generation_id')}")
    if no_select:
        click.echo("Selector unchanged: pass no --no-select to make it current.")


@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def load(config_path):
    """Optionally pre-build the dense zarr cache from raw artifacts."""
    from .cli.load_adata import load_dense_cache

    load_dense_cache(config_path)


##########################################


####### Preprocessing ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def preprocess(config_path):
    """Preprocessing."""
    from .cli.recipes import run_experiment_target

    run_experiment_target(config_path, "preprocess")


##########################################


####### Spatial ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def spatial(config_path):
    """Spatial signal analysis"""
    from .cli.recipes import run_experiment_target

    run_experiment_target(config_path, "spatial")


##########################################


####### HMM ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def hmm(config_path):
    """HMM feature annotations and plotting"""
    from .cli.recipes import run_experiment_target

    run_experiment_target(config_path, "hmm")


##########################################


####### Latent ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def latent(config_path):
    """Latent representations of signal"""
    from .cli.recipes import run_experiment_target

    run_experiment_target(config_path, "latent")


##########################################


####### Variant ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def variant(config_path):
    """Deprecated alias for integrated preprocess variant reporting."""
    from .cli.variant_adata import VARIANT_DEPRECATION_MESSAGE, variant_adata

    click.echo(f"DEPRECATED: {VARIANT_DEPRECATION_MESSAGE}", err=True)
    variant_adata(config_path)


##########################################


####### Chimeric ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def chimeric(config_path):
    """Finding putative PCR chimeras"""
    from .cli.chimeric_adata import chimeric_adata

    chimeric_adata(config_path)


##########################################


####### Recipes ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def full(config_path):
    """Workflow: raw preprocess spatial hmm, then latent by default."""
    full_flow(config_path)


##########################################


####### Read-only semantic planning ###########
@experiment_group.command("plan")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--target",
    type=click.Choice(["raw", "preprocess", "variant", "spatial", "hmm", "latent", "full"]),
    default="full",
    show_default=True,
    help="Experiment target to plan without executing it.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the stable machine-readable plan schema.",
)
@click.option(
    "--upgrade-impact",
    is_flag=True,
    help="Group installed-code impact and report historical recompute cost.",
)
def experiment_plan(config_path, target: str, as_json: bool, upgrade_impact: bool):
    """Explain experiment compatibility and required recomputation without writes."""
    from .pipeline.experiment_graph import (
        format_experiment_plan,
        plan_experiment,
        plan_experiment_upgrade_impact,
    )
    from .pipeline.upgrade_impact import format_upgrade_impact

    if upgrade_impact:
        report = plan_experiment_upgrade_impact(config_path, target)
        click.echo(report.to_json() if as_json else format_upgrade_impact(report))
    else:
        plan = plan_experiment(config_path, target)
        click.echo(plan.to_json() if as_json else format_experiment_plan(plan))


##########################################


####### Engine-facing workflow contract ###########
@experiment_group.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--target",
    type=click.Choice(["raw", "preprocess", "variant", "spatial", "hmm", "latent", "full"]),
    default="full",
    show_default=True,
    help="Experiment target to execute.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Exclusive task-local root for every generated artifact.",
)
@click.option(
    "--input",
    "input_path",
    default=None,
    help="Override input_data_path with a staged local path or file:// URI.",
)
@click.option(
    "--fasta",
    "fasta_path",
    default=None,
    help="Override the reference FASTA with a staged local path or file:// URI.",
)
@click.option(
    "--result-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Result path inside OUTPUT_ROOT (default: workflow_result.json).",
)
@click.option("--cpus", type=click.IntRange(min=1), default=None, help="Task-local CPU ceiling.")
@click.option(
    "--memory-gb",
    type=click.FloatRange(min=0.001),
    default=None,
    help="Task-local memory ceiling in GiB.",
)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=None,
    help="Task-local accelerator decision, bounded by config and availability.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail when a requested optional external tool or report cannot run.",
)
def experiment_run(
    config_path,
    target,
    output_root,
    input_path,
    fasta_path,
    result_json,
    cpus,
    memory_gb,
    accelerator,
    strict,
):
    """Execute one experiment with a stable workflow result contract."""
    from .cli.workflow_contract import WorkflowContractError, run_experiment_workflow

    try:
        path = run_experiment_workflow(
            config_path,
            target=target,
            output_root=output_root,
            input_path=input_path,
            fasta_path=fasta_path,
            result_json=result_json,
            cpus=cpus,
            memory_gb=memory_gb,
            accelerator=accelerator,
            strict=strict,
        )
    except WorkflowContractError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(path)


@experiment_group.command("validate")
@click.argument("output_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--result-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Result path inside OUTPUT_ROOT (default: workflow_result.json).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit structured validation JSON.")
def experiment_validate(output_root, result_json, as_json):
    """Validate a completed or relocated workflow output without writing."""
    import json

    from .cli.workflow_contract import WorkflowContractError, validate_workflow_output

    try:
        validation = validate_workflow_output(output_root, result_json=result_json)
    except WorkflowContractError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(validation, sort_keys=True, separators=(",", ":"), indent=2))
    elif validation["valid"]:
        click.echo("Workflow output is valid.")
    else:
        for issue in validation["issues"]:
            click.echo(f"{issue['code']}: {issue['message']}", err=True)
    if not validation["valid"]:
        raise click.exceptions.Exit(1)


_GENERATION_STAGES = ("raw", "preprocess", "variant", "spatial", "hmm", "latent", "chimeric")


class _GenerationGroup(click.Group):
    """Preserve listing flags after OUTPUT_ROOT while supporting subcommands."""

    _LISTING_FLAGS = frozenset({"--json", "--size"})

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Click groups stop parsing parent options at the first argument so
        # their subcommand can own the remaining options. Before this command
        # became a group, both ``ROOT --json`` and ``ROOT --size`` were public
        # syntax. Normalize only invocations with no subcommand; subcommand
        # arguments and options retain Click's ordinary parsing behavior.
        if not any(argument in self.commands for argument in args):
            listing_flags = [argument for argument in args if argument in self._LISTING_FLAGS]
            if listing_flags:
                args = [
                    *listing_flags,
                    *(argument for argument in args if argument not in self._LISTING_FLAGS),
                ]
        return super().parse_args(ctx, args)


@experiment_group.group(
    "generations",
    invoke_without_command=True,
    cls=_GenerationGroup,
)
@click.argument("output_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--size",
    "include_size",
    is_flag=True,
    help="Total each generation's bytes on disk (slower on large stores).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the stable machine-readable schema.")
@click.pass_context
def experiment_generations_cmd(ctx, output_root, include_size, as_json):
    """List and manage retention metadata for immutable generations."""
    from .cli.generations import experiment_generations, render_json, render_table

    ctx.obj = {"output_root": output_root}
    if ctx.invoked_subcommand is None:
        records = experiment_generations(output_root, include_size=include_size)
        click.echo(render_json(records) if as_json else render_table(records))


@experiment_generations_cmd.command("pin")
@click.argument("stage", type=click.Choice(_GENERATION_STAGES, case_sensitive=False))
@click.argument("generation_id")
@click.option("--reason", required=True, help="Durable reason this generation must survive.")
@click.pass_context
def experiment_generations_pin_cmd(ctx, stage, generation_id, reason):
    """Add a retention reason without modifying the generation manifest."""
    from .cli.generations import pin_experiment_generation
    from .informatics.generation_retention import GenerationRetentionError

    try:
        entry = pin_experiment_generation(
            ctx.obj["output_root"],
            stage,
            generation_id,
            reason=reason,
        )
    except GenerationRetentionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Pinned {stage}/{generation_id} with {len(entry.reasons)} retention reason(s).")


@experiment_generations_cmd.command("unpin")
@click.argument("stage", type=click.Choice(_GENERATION_STAGES, case_sensitive=False))
@click.argument("generation_id")
@click.option("--reason", default=None, help="Remove one exact retention reason.")
@click.option(
    "--all-reasons",
    is_flag=True,
    help="Remove every retention reason for this generation.",
)
@click.pass_context
def experiment_generations_unpin_cmd(ctx, stage, generation_id, reason, all_reasons):
    """Remove an explicit retention reason or all reasons."""
    from .cli.generations import unpin_experiment_generation
    from .informatics.generation_retention import GenerationRetentionError

    if (reason is None) == (not all_reasons):
        raise click.UsageError("choose exactly one of --reason or --all-reasons")
    try:
        remaining = unpin_experiment_generation(
            ctx.obj["output_root"],
            stage,
            generation_id,
            reason=None if all_reasons else reason,
        )
    except GenerationRetentionError as exc:
        raise click.ClickException(str(exc)) from exc
    if remaining is None:
        click.echo(f"Unpinned {stage}/{generation_id}.")
    else:
        click.echo(
            f"Removed one reason from {stage}/{generation_id}; "
            f"{len(remaining.reasons)} reason(s) remain."
        )


@experiment_generations_cmd.command("prune")
@click.option(
    "--stage",
    "stages",
    multiple=True,
    type=click.Choice(_GENERATION_STAGES, case_sensitive=False),
    help="Restrict planning to a stage; repeat for multiple stages.",
)
@click.option(
    "--keep-last",
    type=click.IntRange(min=0),
    default=None,
    help="Protect the newest N generations of each selected stage.",
)
@click.option(
    "--older-than",
    default=None,
    metavar="ISO_TIMESTAMP",
    help="Consider only generations older than this ISO-8601 timestamp.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the versioned pruning plan.")
@click.pass_context
def experiment_generations_prune_cmd(ctx, stages, keep_last, older_than, as_json):
    """Plan retention pruning without deleting anything."""
    from .cli.generations import plan_experiment_prune, render_prune_json, render_prune_table
    from .informatics.generation_pruning import GenerationPruneError

    try:
        plan = plan_experiment_prune(
            ctx.obj["output_root"],
            keep_last=keep_last,
            older_than=older_than,
            stages=stages,
        )
    except GenerationPruneError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_prune_json(plan) if as_json else render_prune_table(plan))


##########################################


####### batch command ###########
@experiment_group.command()
@click.argument(
    "task",
    type=click.Choice(
        ["raw", "load", "preprocess", "spatial", "variant", "hmm", "latent", "full"],
        case_sensitive=False,
    ),
)
@click.argument(
    "config_table",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--column",
    "-c",
    default="config_path",
    show_default=True,
    help="Column name containing config paths (ignored for plain TXT).",
)
@click.option(
    "--sep",
    default=None,
    help="Field separator: default auto-detect (.tsv -> '\\t', .csv -> ',', others treated as TXT).",
)
@click.option(
    "--summary",
    "summary_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Machine-readable result path (default: beside CONFIG_TABLE).",
)
def batch(
    task,
    config_table: Path,
    column: str,
    sep: str | None,
    summary_path: Path | None,
):
    """
    Run a TASK (raw, load, preprocess, spatial, variant, hmm, latent, full) on multiple CONFIG_PATHs
    listed in a CSV/TSV or plain TXT file.

    Plain text format: one config path per line, no header.
    """

    # ----------------------------
    # Decide file type
    # ----------------------------
    suffix = config_table.suffix.lower()

    # TXT mode → each line is a config path
    if suffix in {".txt", ".list"}:
        paths = []
        with config_table.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(Path(line).expanduser())

        if not paths:
            raise click.ClickException(f"No config paths found in text file: {config_table}")

        config_paths = paths

    else:
        # CSV / TSV mode
        # auto-detect separator if not provided
        if sep is None:
            if suffix in {".tsv", ".tab"}:
                sep = "\t"
            else:
                sep = ","

        try:
            df = pd.read_csv(config_table, sep=sep, dtype=str)
        except Exception as e:
            raise click.ClickException(f"Failed to read table {config_table}: {e}") from e

        if df.empty:
            raise click.ClickException(f"Config table is empty: {config_table}")

        # If table has no header or only one column, treat it as raw paths
        if df.shape[1] == 1 and column not in df.columns:
            # re-read as headerless single-column list, so we don't drop the first path
            try:
                df = pd.read_csv(
                    config_table,
                    sep=sep,
                    header=None,
                    names=[column],
                    dtype=str,
                )
            except Exception as e:
                raise click.ClickException(
                    f"Failed to read {config_table} as headerless list: {e}"
                ) from e

            config_series = df[column]
        else:
            if column not in df.columns:
                raise click.ClickException(
                    f"Column '{column}' not found in {config_table}. "
                    f"Available columns: {', '.join(df.columns)}"
                )
            config_series = df[column]

        config_paths = config_series.dropna().map(str).map(lambda p: Path(p).expanduser()).tolist()

    # ----------------------------
    # Validate config paths
    # ----------------------------
    if not config_paths:
        raise click.ClickException("No config paths found.")

    # ----------------------------
    # Map task to function
    # ----------------------------
    task = task.lower()

    def _raw(cfg_path: str):
        from .cli.recipes import run_experiment_target

        return run_experiment_target(cfg_path, "raw")

    def _load(cfg_path: str):
        from .cli.load_adata import load_dense_cache

        return load_dense_cache(cfg_path)

    def _preprocess(cfg_path: str):
        from .cli.recipes import run_experiment_target

        return run_experiment_target(cfg_path, "preprocess")

    def _spatial(cfg_path: str):
        from .cli.recipes import run_experiment_target

        return run_experiment_target(cfg_path, "spatial")

    def _variant(cfg_path: str):
        from .cli.variant_adata import variant_adata

        return variant_adata(cfg_path)

    def _hmm(cfg_path: str):
        from .cli.recipes import run_experiment_target

        return run_experiment_target(cfg_path, "hmm")

    def _latent(cfg_path: str):
        from .cli.recipes import run_experiment_target

        return run_experiment_target(cfg_path, "latent")

    def _full(cfg_path: str):
        from .cli.recipes import full_flow

        return full_flow(cfg_path)

    task_funcs = {
        "raw": _raw,
        "load": _load,
        "preprocess": _preprocess,
        "spatial": _spatial,
        "variant": _variant,
        "hmm": _hmm,
        "latent": _latent,
        "full": _full,
    }

    func = task_funcs[task]

    click.echo(f"Running task '{task}' on {len(config_paths)} config paths from {config_table}")

    if summary_path is None:
        summary_path = config_table.with_name(f"{config_table.stem}.{task}.batch-summary.json")

    from .cli.batch import run_batch

    summary = run_batch(
        task,
        config_paths,
        func,
        config_table=config_table,
        summary_path=summary_path,
        emit=click.echo,
    )
    click.echo(f"Batch summary: {summary_path}")
    if summary["failed"]:
        raise click.ClickException(
            f"Batch completed with {summary['failed']} failure(s) out of {summary['total']} configs"
        )
    click.echo("Batch processing complete.")


##########################################


@cli.command("versions")
@click.option(
    "--tool",
    "tools",
    type=click.Choice(
        [
            "bedGraphToBigWig",
            "bedtools",
            "dorado",
            "gzip",
            "minimap2",
            "modkit",
            "multiqc",
            "pod5",
            "samtools",
        ]
    ),
    multiple=True,
    help="Include one external tool version; repeat for multiple tools.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit stable machine-readable JSON.")
def versions_cmd(tools, as_json):
    """Report stable smftools, Python, and requested external-tool versions."""
    import json

    from .cli.workflow_contract import software_versions

    versions = software_versions(tools=tuple(tools))
    if as_json:
        click.echo(json.dumps(versions, sort_keys=True, separators=(",", ":"), indent=2))
        return
    click.echo(f"smftools {versions['smftools']}")
    click.echo(f"Python {versions['python']}")
    for name, record in versions["external_tools"].items():
        click.echo(f"{name}: {record.get('version') or 'unavailable'}")


##########################################


####### concatenate command ###########
@experiment_group.command("concatenate")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--recompute-pp-vars",
    is_flag=True,
    help="Recompute calculate_coverage and append_base_context after concatenation.",
)
@click.option(
    "--input-dir",
    "-d",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Override concatenate_input_dir from config.",
)
@click.option(
    "--csv-path",
    "-c",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Override concatenate_csv_path from config.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Explicit output path (overrides stage auto-detection).",
)
def concatenate_cmd(
    config_path: Path,
    recompute_pp_vars: bool,
    input_dir: Path | None,
    csv_path: Path | None,
    output_path: Path | None,
):
    """
    Concatenate multiple .h5ad files into a single output file.

    Reads concatenation parameters from an experiment config CSV.
    Input source (directory or CSV of paths) is configured via
    concatenate_input_dir / concatenate_csv_path in the config, or
    overridden with --input-dir / --csv-path.

    Output path is auto-detected from the pipeline stage of the input
    filenames (e.g. *_variant.h5ad → variant output directory). Use
    --output-path to override.

    Example:

        smftools experiment concatenate experiment_config.csv

        smftools experiment concatenate experiment_config.csv --recompute-pp-vars

        smftools experiment concatenate experiment_config.csv --input-dir ./variant_h5ads/
    """
    from .cli.helpers import load_experiment_config
    from .readwrite import concatenate_h5ads

    try:
        cfg = load_experiment_config(str(config_path))

        # Resolve input source: CLI flags override config values
        effective_input_dir = input_dir or (
            Path(cfg.concatenate_input_dir) if cfg.concatenate_input_dir else None
        )
        effective_csv_path = csv_path or (
            Path(cfg.concatenate_csv_path) if cfg.concatenate_csv_path else None
        )

        if effective_input_dir and effective_csv_path:
            raise click.ClickException(
                "Provide only ONE of --input-dir / concatenate_input_dir or "
                "--csv-path / concatenate_csv_path."
            )

        if not effective_input_dir and not effective_csv_path:
            raise click.ClickException(
                "No input source specified. Set concatenate_input_dir or "
                "concatenate_csv_path in the config, or use --input-dir / --csv-path."
            )

        # Determine whether to recompute: CLI flag OR config value
        do_recompute = recompute_pp_vars or cfg.concatenate_recompute_pp_vars

        # Use a placeholder output_path when auto-detection is expected
        effective_output_path = output_path or Path("concatenated_output.h5ad.gz")

        out = concatenate_h5ads(
            output_path=effective_output_path,
            input_dir=effective_input_dir,
            csv_path=effective_csv_path,
            csv_column=cfg.concatenate_csv_column,
            file_suffixes=tuple(cfg.concatenate_file_suffixes),
            delete_inputs=cfg.concatenate_delete_inputs,
            restore_backups=cfg.concatenate_restore_backups,
            recompute_pp_vars=do_recompute,
            config_path=config_path,
        )
        click.echo(f"Concatenated file written to: {out}")

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e


##########################################


####### subsample pod5 command ###########
@cli.command("subsample-pod5")
@click.argument(
    "pod5_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--read-names",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Text file with one read_id per line.",
)
@click.option(
    "--n-reads",
    "-n",
    type=int,
    default=None,
    help="Randomly subsample N reads.",
)
@click.option(
    "--outdir",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output directory for subsampled POD5.",
)
def subsample_pod5_cmd(pod5_path, read_names, n_reads, outdir):
    """
    Subsample POD5 file(s) by read ID list or random sampling.
    """
    from .informatics.pod5_functions import subsample_pod5

    # --- Validate mutually exclusive options ---
    if (read_names is None and n_reads is None) or (read_names and n_reads):
        raise click.UsageError("You must specify exactly ONE of --read-names or --n-reads.")

    outdir.mkdir(parents=True, exist_ok=True)

    subsample_arg = str(read_names) if read_names else n_reads

    subsample_pod5(
        pod5_path=str(pod5_path),
        read_name_path=subsample_arg,
        output_directory=str(outdir),
    )


##########################################


####### Project-level cross-experiment catalog ###########
@cli.group("project")
def project_group():
    """Register experiments into a project and query/analyze across them."""


@project_group.command("init")
@click.argument("project_dir", type=click.Path(path_type=Path))
@click.option(
    "--name", default=None, help="Project name used in scaffolded docs (default: directory name)."
)
def project_init_cmd(project_dir: Path, name):
    """Initialize a project directory + registry, plus starter docs/dirs.

    Creates registry.json, sets/, project_scripts/, project_outputs/, and starter
    README.md/AGENTS.md/CLAUDE.md/PLAN.md/project.yaml files. Safe to re-run --
    only ever fills in what's missing, never overwrites existing files.
    """
    from .cli.project_cmd import project_init

    registry_path, scaffolded = project_init(project_dir, name=name)
    click.echo(f"Initialized project registry: {registry_path}")
    for path in scaffolded:
        click.echo(f"  created {path}")


@project_group.command("add")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("experiment_dir", type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.option(
    "--id",
    "experiment_id",
    default=None,
    help="Explicit experiment id; for modern runs it must match the manifest and directory.",
)
@click.option("--name", default=None, help="Friendly experiment name.")
@click.option(
    "--stage",
    default=None,
    help=(
        "Pipeline stage this registration represents (raw, preprocess, spatial, hmm, "
        "latent, variant, chimeric). Only meaningful when EXPERIMENT_DIR is a legacy "
        "monolithic .h5ad/.h5ad.gz file; otherwise every stage is auto-discovered and "
        "this is ignored. Omit to infer from the legacy file's name."
    ),
)
def project_add_cmd(project_dir: Path, experiment_dir: Path, experiment_id, name, stage):
    """Register EXPERIMENT_DIR into PROJECT_DIR (by pointer; append-only).

    EXPERIMENT_DIR may be a run directory (auto-discovers every pipeline stage
    found under it) or a single legacy monolithic .h5ad/.h5ad.gz file from
    before the partitioned-store pipeline (use --stage to name which stage it
    represents; the source file is only ever read, never modified).
    """
    from .cli.project_cmd import project_add

    try:
        exp_id, entry, conflicts = project_add(
            project_dir,
            experiment_dir,
            experiment_id=experiment_id,
            name=name,
            stage=stage,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Registered '{exp_id}' ({entry['modality']}, {entry['n_reads']} reads, "
        f"{len(entry['references'])} references)"
    )
    for warning in conflicts:
        click.echo(f"  WARNING: {warning}")


@project_group.command("remove")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("experiment_id")
def project_remove_cmd(project_dir: Path, experiment_id: str):
    """Mark an experiment inactive in the project."""
    from .cli.project_cmd import project_remove

    project_remove(project_dir, experiment_id)
    click.echo(f"Removed '{experiment_id}' (marked inactive)")


@project_group.command("list")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def project_list_cmd(project_dir: Path):
    """List registered experiments and harmonized references."""
    from .cli.project_cmd import project_list

    experiments, references = project_list(project_dir)
    click.echo(f"{len(experiments)} experiment(s):")
    for entry in experiments:
        stages = ",".join(sorted(entry.get("spines", {})))
        click.echo(
            f"  {entry['id']}  ({entry['modality']}, {entry['n_reads']} reads, "
            f"stages: {stages})  {entry['path']}"
        )
    if not references.empty:
        n_canon = references["canonical_reference"].nunique()
        click.echo(f"{n_canon} canonical reference(s) across the project.")


@project_group.group("rebasecall")
def project_rebasecall_group():
    """Plan selective POD5 re-basecalling across a project's experiments."""


@project_rebasecall_group.command("plan")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "request_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--set",
    "set_name",
    default=None,
    help="Restrict the fan-out to a named project set.",
)
@click.option(
    "--experiment",
    "experiment_ids",
    multiple=True,
    help="Restrict the fan-out to specific experiment ids; repeat per experiment.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit stable machine-readable JSON.")
def project_rebasecall_plan_cmd(
    project_dir: Path,
    request_path: Path,
    set_name: str | None,
    experiment_ids: tuple[str, ...],
    as_json: bool,
):
    """Report what one request would do to each selected experiment, writing nothing."""
    import json

    from .pipeline.rebasecall_project import (
        format_project_rebasecall_plan,
        plan_project_rebasecall,
    )
    from .pipeline.rebasecall_request import load_rebasecall_request

    try:
        plan = plan_project_rebasecall(
            project_dir,
            load_rebasecall_request(request_path),
            experiments=list(experiment_ids) or None,
            set_name=set_name,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    if as_json:
        click.echo(json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"), indent=2))
        return
    click.echo(format_project_rebasecall_plan(plan))


@project_group.group("analyses")
def project_analyses_group():
    """Inspect project-owned analysis caches without loading their results."""


@project_analyses_group.command("list")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--stale",
    "stale_only",
    is_flag=True,
    help="Show only stale or invalid caches that require attention.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the stable machine-readable schema.")
def project_analyses_list_cmd(project_dir: Path, stale_only: bool, as_json: bool):
    """List periodicity and embedding caches, their code identity, and size."""
    import json

    from .project.analysis_inventory import analysis_cache_inventory

    inventory = analysis_cache_inventory(project_dir, stale_only=stale_only)
    if as_json:
        click.echo(json.dumps(inventory, sort_keys=True, separators=(",", ":"), indent=2))
        return

    entries = inventory["entries"]
    if not entries:
        message = (
            "No stale or invalid project analysis caches found."
            if stale_only
            else "No project analysis caches found."
        )
        click.echo(message)
        return
    click.echo("STATUS   ANALYSIS     SIZE (bytes)  SCOPE                         CACHE")
    for entry in entries:
        scope = (
            f"{entry['experiment_id']}/{entry['reference_strand']}/{entry['sample']}"
            if entry["scope"] == "partition"
            else str(entry["set_label"])
        )
        reasons = ",".join(entry["reasons"]) or "-"
        size = "?" if entry["size_bytes"] is None else str(entry["size_bytes"])
        click.echo(
            f"{entry['status']:<8} {entry['analysis']:<12} {size:>12}  "
            f"{scope:<29} {entry['cache_path']}"
        )
        if entry["status"] != "current":
            click.echo(f"  reason: {reasons}")


@project_group.command("generations")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--size",
    "include_size",
    is_flag=True,
    help="Total each generation's bytes on disk (slower on large stores).",
)
@click.option(
    "--project-only",
    is_flag=True,
    help="List only project-owned generations, skipping registered experiments.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the stable machine-readable schema.")
def project_generations_cmd(project_dir: Path, include_size, project_only, as_json):
    """List published immutable generations across a project, without writing."""
    from .cli.generations import project_generations, render_json, render_table

    records = project_generations(
        project_dir,
        include_size=include_size,
        include_experiments=not project_only,
    )
    click.echo(render_json(records) if as_json else render_table(records))


@project_group.command("plan")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument(
    "target",
    type=click.Choice(("selection", "materialization", "sample-analysis", "embedding")),
)
@click.argument("canonical_reference")
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option("--modality", default=None, help="Restrict to a modality.")
@click.option(
    "--experiment",
    "experiments",
    multiple=True,
    help="Restrict to an experiment ID; repeat for multiple experiments.",
)
@click.option("--stage", default=None, help="Select one experiment pipeline stage.")
@click.option("--start", type=int, default=None, help="Genomic window start (with --end).")
@click.option("--end", type=int, default=None, help="Genomic window end (with --start).")
@click.option("--layers", default=None, help="Comma-separated materialization layer subset.")
@click.option("--read-metrics", is_flag=True, help="Include spatial per-read outputs.")
@click.option("--partitioned", is_flag=True, help="Plan partitioned materialization.")
@click.option("--json", "as_json", is_flag=True, help="Emit stable machine-readable JSON.")
@click.option(
    "--upgrade-impact",
    is_flag=True,
    help="Group installed-code impact; unavailable historical costs remain explicit.",
)
def project_plan_cmd(
    project_dir,
    target,
    canonical_reference,
    set_name,
    modality,
    experiments,
    stage,
    start,
    end,
    layers,
    read_metrics,
    partitioned,
    as_json,
    upgrade_impact,
):
    """Explain reuse and recomputation for a project analysis without writing."""
    from .cli.project_cmd import project_plan, project_upgrade_impact
    from .pipeline.project_graph import format_project_plan
    from .pipeline.upgrade_impact import format_upgrade_impact

    layer_list = None if layers is None else [item for item in layers.split(",") if item]
    planner = project_upgrade_impact if upgrade_impact else project_plan
    result = planner(
        project_dir,
        target,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layer_list,
        read_metrics=read_metrics,
        partitioned=partitioned,
    )
    if upgrade_impact:
        click.echo(result.to_json() if as_json else format_upgrade_impact(result))
    else:
        click.echo(result.to_json() if as_json else format_project_plan(result))


# Which target each `project run` option applies to. `run` is the engine-facing
# entry point for every executable project product, so it accepts the union of
# their options and rejects the ones that do not apply -- silently ignoring a
# flag would publish a result that does not describe what was requested.
_PROJECT_RUN_TARGET_OPTIONS = {
    "materialization": {"layers", "read_metrics", "allow_large", "partitioned"},
    "sample-analysis": {"layer", "method", "force_recompute"},
    "embedding": {
        "layer",
        "feature_kind",
        "leiden_resolution",
        "n_neighbors",
        "min_reads",
        "random_state",
        "force_recompute",
        "trust_local_models",
    },
}
_PROJECT_RUN_OPTION_FLAGS = {
    "layers": "--layers",
    "read_metrics": "--read-metrics",
    "allow_large": "--allow-large",
    "partitioned": "--partitioned",
    "layer": "--layer",
    "method": "--method",
    "force_recompute": "--force-recompute",
    "feature_kind": "--feature-kind",
    "leiden_resolution": "--leiden-resolution",
    "n_neighbors": "--n-neighbors",
    "min_reads": "--min-reads",
    "random_state": "--random-state",
    "trust_local_models": "--trust-local-models",
}


def _reject_project_run_options(context, target: str) -> None:
    """Fail when an explicitly passed option does not apply to *target*."""
    from click.core import ParameterSource

    allowed = _PROJECT_RUN_TARGET_OPTIONS[target]
    offending = sorted(
        flag
        for name, flag in _PROJECT_RUN_OPTION_FLAGS.items()
        if name not in allowed and context.get_parameter_source(name) is ParameterSource.COMMANDLINE
    )
    if offending:
        raise click.ClickException(
            f"{', '.join(offending)} do(es) not apply to --target {target}; "
            f"applicable options are: "
            f"{', '.join(sorted(_PROJECT_RUN_OPTION_FLAGS[name] for name in allowed))}"
        )


@project_group.command("run")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("canonical_reference")
@click.option(
    "--target",
    type=click.Choice(["materialization", "sample-analysis", "embedding"]),
    default="materialization",
    show_default=True,
    help=(
        "Project product to execute. `selection` is a planning-only dependency of "
        "these three and has no artifact of its own."
    ),
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Exclusive task-local root for the artifact and result contract.",
)
@click.option(
    "--output-name",
    default=None,
    help="Artifact name inside OUTPUT_ROOT (default depends on the target).",
)
@click.option(
    "--result-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Result path inside OUTPUT_ROOT (default: workflow_result.json).",
)
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option("--modality", default=None, help="Restrict to a modality.")
@click.option(
    "--experiment",
    "experiments",
    multiple=True,
    help="Restrict to an experiment ID; repeat for multiple experiments.",
)
@click.option("--stage", default=None, help="Select one experiment pipeline stage.")
@click.option("--start", type=int, default=None, help="Genomic window start (with --end).")
@click.option("--end", type=int, default=None, help="Genomic window end (with --start).")
@click.option("--layers", default=None, help="[materialization] Comma-separated layer subset.")
@click.option(
    "--read-metrics", is_flag=True, help="[materialization] Include spatial per-read outputs."
)
@click.option(
    "--allow-large",
    is_flag=True,
    help="[materialization] Acknowledge the pooled-object soft limit.",
)
@click.option("--partitioned", is_flag=True, help="[materialization] Write bounded Zarr parts.")
@click.option(
    "--layer", default=None, help="[sample-analysis, embedding] Layer to analyze (default: X)."
)
@click.option(
    "--method", default="direct", show_default=True, help="[sample-analysis] Periodicity method."
)
@click.option(
    "--feature-kind",
    type=click.Choice(["raw", "acf"]),
    default="raw",
    show_default=True,
    help="[embedding] Feature construction.",
)
@click.option(
    "--leiden-resolution",
    type=float,
    default=0.5,
    show_default=True,
    help="[embedding] Leiden resolution.",
)
@click.option(
    "--n-neighbors",
    type=int,
    default=15,
    show_default=True,
    help="[embedding] Neighborhood size.",
)
@click.option(
    "--min-reads", type=int, default=10, show_default=True, help="[embedding] Minimum reads to fit."
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    show_default=True,
    help="[embedding] Deterministic seed.",
)
@click.option(
    "--force-recompute",
    is_flag=True,
    help="[sample-analysis, embedding] Recompute instead of reusing cached results.",
)
@click.option(
    "--trust-local-models",
    is_flag=True,
    help=(
        "[embedding] Permit loading this project's persisted estimator pickles, which "
        "extending an existing embedding requires. Unpickling executes code from those files."
    ),
)
@click.option("--cpus", type=click.IntRange(min=1), default=None, help="Task-local CPU ceiling.")
@click.option(
    "--memory-gb",
    type=click.FloatRange(min=0.001),
    default=None,
    help="Task-local memory ceiling in GiB.",
)
@click.option(
    "--memory-percent",
    type=click.FloatRange(min=0.001, max=100.0),
    default=60.0,
    show_default=True,
    help="Task-local memory ceiling as a percentage of physical memory.",
)
@click.pass_context
def project_run_cmd(
    context,
    project_dir,
    canonical_reference,
    target,
    output_root,
    output_name,
    result_json,
    set_name,
    modality,
    experiments,
    stage,
    start,
    end,
    layers,
    read_metrics,
    allow_large,
    partitioned,
    layer,
    method,
    feature_kind,
    leiden_resolution,
    n_neighbors,
    min_reads,
    random_state,
    force_recompute,
    trust_local_models,
    cpus,
    memory_gb,
    memory_percent,
):
    """Execute one project product with the stable workflow contract."""
    from .cli.workflow_contract import (
        WorkflowContractError,
        run_project_embedding_workflow,
        run_project_materialization_workflow,
        run_project_sample_analysis_workflow,
    )

    _reject_project_run_options(context, target)
    shared = {
        "output_root": output_root,
        "output_name": output_name,
        "result_json": result_json,
        "set_name": set_name,
        "modality": modality,
        "experiments": experiments,
        "stage": stage,
        "start": start,
        "end": end,
        "cpus": cpus,
        "memory_gb": memory_gb,
        "memory_percent": memory_percent,
    }
    try:
        if target == "materialization":
            path = run_project_materialization_workflow(
                project_dir,
                canonical_reference,
                layers=None if layers is None else [item for item in layers.split(",") if item],
                read_metrics=read_metrics,
                allow_large=allow_large,
                partitioned=partitioned,
                **shared,
            )
        elif target == "sample-analysis":
            path = run_project_sample_analysis_workflow(
                project_dir,
                canonical_reference,
                layer=layer,
                method=method,
                force_recompute=force_recompute,
                **shared,
            )
        else:
            path = run_project_embedding_workflow(
                project_dir,
                canonical_reference,
                layer=layer,
                feature_kind=feature_kind,
                leiden_resolution=leiden_resolution,
                n_neighbors=n_neighbors,
                min_reads=min_reads,
                random_state=random_state,
                force_recompute=force_recompute,
                trust_local_models=trust_local_models,
                **shared,
            )
    except WorkflowContractError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(path)


@project_group.command("sample-analysis")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("canonical_reference")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Exclusive task-local root for every generated artifact.",
)
@click.option(
    "--output-name",
    default=None,
    help="Result table name inside OUTPUT_ROOT (default: sample_analysis.parquet).",
)
@click.option(
    "--result-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Result path inside OUTPUT_ROOT (default: workflow_result.json).",
)
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option("--modality", default=None, help="Restrict to a modality.")
@click.option(
    "--experiment",
    "experiments",
    multiple=True,
    help="Restrict to an experiment ID; repeat for multiple experiments.",
)
@click.option("--stage", default=None, help="Select one experiment pipeline stage.")
@click.option("--start", type=int, default=None, help="Genomic window start (with --end).")
@click.option("--end", type=int, default=None, help="Genomic window end (with --start).")
@click.option("--layer", default=None, help="Layer to analyze (default: X).")
@click.option(
    "--method",
    default="direct",
    show_default=True,
    help="Periodicity method: 'direct' single-molecule, or an ACF-intermediate method.",
)
@click.option(
    "--force-recompute",
    is_flag=True,
    help="Recompute each partition instead of reusing its cached per-sample result.",
)
@click.option("--cpus", type=click.IntRange(min=1), default=None, help="Task-local CPU ceiling.")
@click.option(
    "--memory-gb",
    type=click.FloatRange(min=0.001),
    default=None,
    help="Task-local memory ceiling in GiB.",
)
@click.option(
    "--max-memory-percent",
    "memory_percent",
    type=click.FloatRange(min=1, max=100),
    default=60.0,
    show_default=True,
    help="Ceiling as a percentage of available memory.",
)
def project_sample_analysis_cmd(
    project_dir,
    canonical_reference,
    output_root,
    output_name,
    result_json,
    set_name,
    modality,
    experiments,
    stage,
    start,
    end,
    layer,
    method,
    force_recompute,
    cpus,
    memory_gb,
    memory_percent,
):
    """Run per-sample periodicity across a project selection, task-locally."""
    from .cli.workflow_contract import (
        WorkflowContractError,
        run_project_sample_analysis_workflow,
    )

    try:
        path = run_project_sample_analysis_workflow(
            project_dir,
            canonical_reference,
            output_root=output_root,
            output_name=output_name,
            result_json=result_json,
            set_name=set_name,
            modality=modality,
            experiments=experiments,
            stage=stage,
            start=start,
            end=end,
            layer=layer,
            method=method,
            force_recompute=force_recompute,
            cpus=cpus,
            memory_gb=memory_gb,
            memory_percent=memory_percent,
        )
    except WorkflowContractError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(path)


@project_group.command("embedding")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("canonical_reference")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Exclusive task-local root for every generated artifact.",
)
@click.option(
    "--output-name",
    default=None,
    help="Coordinate table name inside OUTPUT_ROOT (default: embedding.parquet).",
)
@click.option(
    "--result-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Result path inside OUTPUT_ROOT (default: workflow_result.json).",
)
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option("--modality", default=None, help="Restrict to a modality.")
@click.option(
    "--experiment",
    "experiments",
    multiple=True,
    help="Restrict to an experiment ID; repeat for multiple experiments.",
)
@click.option("--stage", default=None, help="Select one experiment pipeline stage.")
@click.option("--start", type=int, default=None, help="Genomic window start (with --end).")
@click.option("--end", type=int, default=None, help="Genomic window end (with --start).")
@click.option("--layer", default=None, help="Layer to embed (default: X).")
@click.option(
    "--feature-kind",
    type=click.Choice(["raw", "acf"]),
    default="raw",
    show_default=True,
    help="Feature construction for the shared space.",
)
@click.option(
    "--leiden-resolution", type=float, default=0.5, show_default=True, help="Leiden resolution."
)
@click.option(
    "--n-neighbors", type=int, default=15, show_default=True, help="UMAP/cluster neighborhood size."
)
@click.option(
    "--min-reads", type=int, default=10, show_default=True, help="Minimum reads required to fit."
)
@click.option("--random-state", type=int, default=42, show_default=True, help="Deterministic seed.")
@click.option(
    "--force-recompute",
    is_flag=True,
    help="Refit from scratch. Required when molecules were removed or their features changed.",
)
@click.option(
    "--trust-local-models",
    is_flag=True,
    help=(
        "Permit loading this project's persisted PCA/UMAP estimator pickles, which "
        "extending an existing embedding requires. Pass this only for a project tree "
        "you trust; unpickling executes code from those files."
    ),
)
@click.option("--cpus", type=click.IntRange(min=1), default=None, help="Task-local CPU ceiling.")
@click.option(
    "--memory-gb",
    type=click.FloatRange(min=0.001),
    default=None,
    help="Task-local memory ceiling in GiB.",
)
@click.option(
    "--max-memory-percent",
    "memory_percent",
    type=click.FloatRange(min=1, max=100),
    default=60.0,
    show_default=True,
    help="Ceiling as a percentage of available memory.",
)
def project_embedding_cmd(
    project_dir,
    canonical_reference,
    output_root,
    output_name,
    result_json,
    set_name,
    modality,
    experiments,
    stage,
    start,
    end,
    layer,
    feature_kind,
    leiden_resolution,
    n_neighbors,
    min_reads,
    random_state,
    force_recompute,
    trust_local_models,
    cpus,
    memory_gb,
    memory_percent,
):
    """Fit or extend one shared project embedding, task-locally."""
    from .cli.workflow_contract import WorkflowContractError, run_project_embedding_workflow

    try:
        path = run_project_embedding_workflow(
            project_dir,
            canonical_reference,
            output_root=output_root,
            output_name=output_name,
            result_json=result_json,
            set_name=set_name,
            modality=modality,
            experiments=experiments,
            stage=stage,
            start=start,
            end=end,
            layer=layer,
            feature_kind=feature_kind,
            leiden_resolution=leiden_resolution,
            n_neighbors=n_neighbors,
            min_reads=min_reads,
            random_state=random_state,
            force_recompute=force_recompute,
            trust_local_models=trust_local_models,
            cpus=cpus,
            memory_gb=memory_gb,
            memory_percent=memory_percent,
        )
    except WorkflowContractError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(path)


@project_group.command("validate")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--result-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Result path inside OUTPUT_ROOT (default: workflow_result.json).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit structured validation JSON.")
def project_validate_cmd(project_dir, output_root, result_json, as_json):
    """Validate project output integrity and current source compatibility."""
    import json

    from .cli.workflow_contract import WorkflowContractError, validate_workflow_output

    try:
        validation = validate_workflow_output(
            output_root,
            result_json=result_json,
            project_dir=project_dir,
        )
    except WorkflowContractError as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(validation, sort_keys=True, separators=(",", ":"), indent=2))
    elif validation["valid"]:
        click.echo("Project workflow output is valid.")
    else:
        for issue in validation["issues"]:
            click.echo(f"{issue['code']}: {issue['message']}", err=True)
    if not validation["valid"]:
        raise click.exceptions.Exit(1)


@project_group.command("materialize")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("canonical_reference")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), required=True, help="Output .h5ad(.gz)."
)
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option("--modality", default=None, help="Restrict to a modality.")
@click.option(
    "--stage",
    default=None,
    help=(
        "Pipeline stage to materialize per experiment (raw/preprocess/spatial/hmm/"
        "variant/chimeric). Latent is available only through export-latent or the "
        "scoped project API. Default: most-derived stage available per "
        "experiment, since a later stage already carries forward earlier stages' data."
    ),
)
@click.option("--start", type=int, default=None, help="Genomic window start (with --end).")
@click.option("--end", type=int, default=None, help="Genomic window end (with --start).")
@click.option(
    "--layers",
    default=None,
    help=(
        "Comma-separated layer subset to pool (e.g. 'C_site_binary'). Strongly "
        "recommended for cross-experiment pools -- the default pools every layer at "
        "full locus, which builds enormous objects. Use '' for X only (no layers)."
    ),
)
@click.option(
    "--read-metrics",
    is_flag=True,
    help="Also attach spatial-stage per-read outputs (autocorrelation, Lomb-Scargle) when available.",
)
@click.option(
    "--allow-large",
    is_flag=True,
    help=(
        "Acknowledge the ~8 GiB pooled-object warning. This never bypasses the resolved "
        "hard memory ceiling."
    ),
)
@click.option(
    "--partitioned",
    is_flag=True,
    help="Write a cataloged directory of bounded Zarr parts instead of one pooled H5AD.",
)
@click.option(
    "--max-memory-gb",
    type=click.FloatRange(min=0.001),
    default=None,
    help="Optional hard project-materialization memory ceiling in GiB.",
)
@click.option(
    "--max-memory-percent",
    type=click.FloatRange(min=0.001, max=100.0),
    default=60.0,
    show_default=True,
    help="Hard project-materialization ceiling as a percentage of physical memory.",
)
def project_materialize_cmd(
    project_dir,
    canonical_reference,
    output,
    set_name,
    modality,
    stage,
    start,
    end,
    layers,
    read_metrics,
    allow_large,
    partitioned,
    max_memory_gb,
    max_memory_percent,
):
    """Pool CANONICAL_REFERENCE across matching experiments into one AnnData.

    Prefer --layers and/or --start/--end. Pooled output is preflighted before allocation;
    --allow-large acknowledges its warning threshold but not the hard memory ceiling.
    Use --partitioned for a cataloged directory of bounded Zarr parts.
    """
    from .cli.project_cmd import project_materialize

    layer_list = None if layers is None else [s for s in layers.split(",") if s]
    out = project_materialize(
        project_dir,
        canonical_reference,
        output,
        set_name=set_name,
        modality=modality,
        stage=stage,
        start=start,
        end=end,
        layers=layer_list,
        read_metrics=read_metrics,
        allow_large=allow_large,
        partitioned=partitioned,
        max_memory_gb=max_memory_gb,
        max_memory_percent=max_memory_percent,
    )
    click.echo(f"Wrote {out}")


@project_group.command("export-latent")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--canonical-reference",
    default=None,
    help="Restrict to one canonical reference UID.",
)
@click.option(
    "--experiment",
    "experiments",
    multiple=True,
    help="Restrict to an experiment ID; repeat for multiple experiments.",
)
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option(
    "--molecule-uid",
    "molecule_uids",
    multiple=True,
    help="Restrict to a molecule UID; repeat for multiple molecules.",
)
@click.option(
    "--analysis-core-id",
    "analysis_core_ids",
    multiple=True,
    help="Restrict to an analysis-core owner; repeat for multiple cores.",
)
@click.option(
    "--representations",
    default=None,
    help="Comma-separated task-local obsm/varm keys to export.",
)
@click.option(
    "--labels",
    default=None,
    help="Comma-separated task-local observation labels to export.",
)
def project_export_latent_cmd(
    project_dir,
    output_dir,
    canonical_reference,
    experiments,
    set_name,
    molecule_uids,
    analysis_core_ids,
    representations,
    labels,
):
    """Export task-local latent results without pooling coordinate owners."""
    from .cli.project_cmd import project_export_latent

    representation_list = (
        None
        if representations is None
        else [value for value in representations.split(",") if value]
    )
    label_list = None if labels is None else [value for value in labels.split(",") if value]
    output = project_export_latent(
        project_dir,
        output_dir,
        canonical_reference=canonical_reference,
        experiments=experiments or None,
        set_name=set_name,
        molecule_uids=molecule_uids or None,
        analysis_core_ids=analysis_core_ids or None,
        representations=representation_list,
        labels=label_list,
    )
    click.echo(f"Wrote {output}")


@project_group.command("sample-store-list")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--experiment-id", default=None, help="Restrict to one experiment.")
def project_sample_store_list_cmd(project_dir: Path, experiment_id):
    """List per-sample-store partitions (Reference_strand x sample) cataloged by project add."""
    from .cli.project_cmd import project_sample_store_list

    partitions = project_sample_store_list(project_dir, experiment_id)
    if not partitions:
        click.echo("No per-sample-store partitions cataloged yet.")
        return
    click.echo(f"{len(partitions)} partition(s):")
    for partition in partitions:
        click.echo(
            f"  {partition['experiment_id']}  {partition['reference_strand']}  "
            f"{partition['sample']}  ({partition['kind']}, {partition['n_reads']} reads)"
        )


##########################################


####### named experiment sets ###########
@project_group.command("add-set")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("name")
@click.option(
    "--experiment",
    "experiments",
    multiple=True,
    help="Experiment ID to include; repeat for multiple. Mutually exclusive with --query.",
)
@click.option(
    "--query",
    default=None,
    help="Saved SQL predicate over the harmonized refs table, e.g. \"modality='direct'\".",
)
@click.option(
    "--allow-unresolved",
    is_flag=True,
    help="Define the set even if it names an unregistered, inactive, or repeated experiment.",
)
def project_add_set_cmd(project_dir: Path, name, experiments, query, allow_unresolved):
    """Define a named experiment set usable as --set by other project commands."""
    from .cli.project_cmd import project_add_set
    from .project.registry import SetMembershipError

    if bool(experiments) == bool(query):
        raise click.ClickException("provide exactly one of --experiment or --query")
    try:
        membership = project_add_set(
            project_dir,
            name,
            experiments=list(experiments) if experiments else None,
            query=query,
            allow_unresolved=allow_unresolved,
        )
    except SetMembershipError as exc:
        raise click.ClickException(
            f"{exc}. Register the experiments first, correct the name, "
            f"or pass --allow-unresolved to define the set anyway."
        ) from exc
    click.echo(f"Defined set '{name}' ({membership.kind}).")
    _echo_set_membership(membership)


@project_group.command("list-sets")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def project_list_sets_cmd(project_dir: Path):
    """List named experiment sets without resolving them."""
    from .cli.project_cmd import project_list_sets

    records = project_list_sets(project_dir)
    if not records:
        click.echo("No named sets defined. Create one with `smftools project add-set`.")
        return
    click.echo(f"{len(records)} set(s):")
    for record in records:
        detail = (
            f"query: {record['query']}"
            if record["kind"] == "query"
            else f"{record['n_declared']} declared experiment(s)"
        )
        click.echo(f"  {record['name']}  ({record['kind']}) {detail}")
    click.echo("Run `smftools project show-set PROJECT_DIR NAME` to resolve one.")


@project_group.command("show-set")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("name")
def project_show_set_cmd(project_dir: Path, name):
    """Show the experiments a named set resolves to, exactly as --set applies it."""
    from .cli.project_cmd import project_show_set

    try:
        membership = project_show_set(project_dir, name)
    except KeyError as exc:
        raise click.ClickException(
            f"no set {name!r} in project; `smftools project list-sets` shows the defined sets"
        ) from exc
    click.echo(f"Set '{membership.name}' ({membership.kind})")
    if membership.query is not None:
        click.echo(f"  query: {membership.query}")
    _echo_set_membership(membership)


@project_group.command("remove-set")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("name")
def project_remove_set_cmd(project_dir: Path, name):
    """Delete a named set. Registered experiments are never affected."""
    from .cli.project_cmd import project_remove_set

    try:
        project_remove_set(project_dir, name)
    except KeyError as exc:
        raise click.ClickException(
            f"no set {name!r} in project; `smftools project list-sets` shows the defined sets"
        ) from exc
    click.echo(f"Removed set '{name}'. No experiment registration was changed.")


def _echo_set_membership(membership) -> None:
    """Print resolved membership plus anything declared that does not resolve."""
    if membership.resolved:
        click.echo(f"  resolves to {len(membership.resolved)} experiment(s):")
        for experiment_id in membership.resolved:
            click.echo(f"    {experiment_id}")
    else:
        click.echo("  resolves to no experiments; --set would select nothing.")
    if membership.missing:
        click.echo(f"  not registered: {', '.join(membership.missing)}", err=True)
    if membership.inactive:
        click.echo(f"  inactive: {', '.join(membership.inactive)}", err=True)
    if membership.duplicates:
        click.echo(f"  listed more than once: {', '.join(membership.duplicates)}", err=True)


##########################################


####### FASTQ export ###########
@experiment_group.command("export-fastq")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--outdir",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output directory for FASTQ files + manifest CSV.",
)
@click.option(
    "--group-by",
    default=None,
    help="obs column to group reads by (default: Sample/Barcode).",
)
@click.option(
    "--allow-unfiltered",
    is_flag=True,
    help="Write all reads when no QC-passed read set is available, instead of raising/skipping.",
)
@click.option(
    "--no-gzip",
    is_flag=True,
    help="Write plain .fastq instead of .fastq.gz.",
)
def export_fastq_experiment_cmd(
    config_path: Path,
    outdir: Path,
    group_by: str | None,
    allow_unfiltered: bool,
    no_gzip: bool,
):
    """Write one FASTQ per barcode of QC-passed reads, for one experiment.

    Reads sequence/quality directly from the raw ragged store; the QC-passed read
    set is resolved from the most complete preprocessing artifact available.

    Example:

        smftools experiment export-fastq experiment_config.csv --outdir ./fastqs
    """
    from .cli.export_fastq import export_fastq_for_experiment

    out = export_fastq_for_experiment(
        str(config_path),
        outdir,
        group_by=group_by,
        allow_unfiltered=allow_unfiltered,
        gzip_output=not no_gzip,
    )
    click.echo(f"Wrote FASTQ export to: {out}")


@project_group.command("export-fastq")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--outdir",
    "-o",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output directory for FASTQ files + manifest CSV.",
)
@click.option(
    "--experiments",
    default=None,
    help="Comma-separated experiment ids to include (default: all active).",
)
@click.option(
    "--allow-unfiltered",
    is_flag=True,
    help="Write all reads when no QC-passed read set is available, instead of raising/skipping.",
)
@click.option(
    "--no-gzip",
    is_flag=True,
    help="Write plain .fastq instead of .fastq.gz.",
)
def export_fastq_project_cmd(
    project_dir: Path,
    outdir: Path,
    experiments: str | None,
    allow_unfiltered: bool,
    no_gzip: bool,
):
    """Write one FASTQ per barcode of QC-passed reads, across every registered experiment.

    Example:

        smftools project export-fastq ./my_project --outdir ./fastqs
    """
    from .cli.export_fastq import export_fastq_for_project

    experiment_list = (
        [item.strip() for item in experiments.split(",") if item.strip()] if experiments else None
    )
    out = export_fastq_for_project(
        project_dir,
        outdir,
        experiments=experiment_list,
        allow_unfiltered=allow_unfiltered,
        gzip_output=not no_gzip,
    )
    click.echo(f"Wrote FASTQ export to: {out}")


@experiment_group.command("export-bundle")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--outdir", "-o", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option(
    "--format",
    "bundle_format",
    type=click.Choice(["fastq", "bam"], case_sensitive=False),
    default="fastq",
    show_default=True,
    help="FASTQ is sequence-only; BAM preserves alignment and auxiliary tags.",
)
@click.option("--group-by", default=None, help="obs column used for FASTQ grouping.")
@click.option("--allow-unfiltered", is_flag=True, help="Allow export without a QC selection.")
@click.option("--no-gzip", is_flag=True, help="Write plain FASTQ instead of FASTQ.gz.")
def export_bundle_experiment_cmd(
    config_path: Path,
    outdir: Path,
    bundle_format: str,
    group_by: str | None,
    allow_unfiltered: bool,
    no_gzip: bool,
):
    """Export a portable, checksummed re-ingestion bundle."""
    from .cli.export_bundle import export_bundle_for_experiment

    out = export_bundle_for_experiment(
        str(config_path),
        outdir,
        bundle_format=bundle_format.lower(),
        group_by=group_by,
        allow_unfiltered=allow_unfiltered,
        gzip_output=not no_gzip,
    )
    click.echo(f"Wrote {bundle_format.upper()} bundle to: {out}")


@project_group.command("export-bundle")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--outdir", "-o", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option(
    "--format",
    "bundle_format",
    type=click.Choice(["fastq", "bam"], case_sensitive=False),
    default="fastq",
    show_default=True,
)
@click.option("--experiments", default=None, help="Comma-separated experiment ids to include.")
@click.option("--allow-unfiltered", is_flag=True, help="Allow export without QC selections.")
@click.option("--no-gzip", is_flag=True, help="Write plain FASTQ instead of FASTQ.gz.")
def export_bundle_project_cmd(
    project_dir: Path,
    outdir: Path,
    bundle_format: str,
    experiments: str | None,
    allow_unfiltered: bool,
    no_gzip: bool,
):
    """Export selected project experiments as a portable bundle."""
    from .cli.export_bundle import export_bundle_for_project

    selected = (
        [item.strip() for item in experiments.split(",") if item.strip()] if experiments else None
    )
    out = export_bundle_for_project(
        project_dir,
        outdir,
        bundle_format=bundle_format.lower(),
        experiments=selected,
        allow_unfiltered=allow_unfiltered,
        gzip_output=not no_gzip,
    )
    click.echo(f"Wrote {bundle_format.upper()} bundle to: {out}")


##########################################


####### Plot current traces ###########
@experiment_group.command("plot-current")
@click.argument("config_path", type=click.Path(exists=True))
def plot_current(config_path):
    """Plot nanopore current traces for specified reads."""
    from .cli.plot_current import plot_current as plot_current_fn

    plot_current_fn(config_path)


####### Volume- and machine-scoped storage operations ###########
@cli.group("data")
def data_group():
    """Machine- and volume-scoped storage operations (portable storage roots).

    Below any single experiment and across all projects -- see PSR in
    dev/plans/in-progress/portable_storage_roots_implementation_plan.md.
    """


@data_group.command("init-volume")
@click.argument("mount", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--label", required=True, help="Human-readable name for this volume.")
@click.option(
    "--kind",
    type=click.Choice(["working", "archive", "backup"], case_sensitive=False),
    default="archive",
    show_default=True,
    help="What role this volume plays.",
)
def data_init_volume_cmd(mount: Path, label: str, kind: str):
    """Stamp MOUNT with a permanent volume identity.

    Writes .smftools-volume.json at the volume root. The stamp is written
    once and never rewritten: re-running this command on an already-stamped
    volume leaves it untouched and reports its existing identity, so a
    drive keeps its volume_id even if it is later relabeled or reattached
    under a different mount point.
    """
    from .cli.data_cmd import data_init_volume

    try:
        stamp, created, warnings = data_init_volume(mount, label=label, kind=kind.lower())
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if created:
        click.echo(
            f"Stamped {mount} as volume {stamp['volume_id']} ({stamp['label']}, {stamp['kind']})"
        )
    else:
        click.echo(
            f"{mount} is already stamped as volume {stamp['volume_id']} "
            f"({stamp['label']}, {stamp['kind']}, created {stamp['created']})"
        )
    for warning in warnings:
        click.echo(f"  WARNING: {warning}")


@data_group.command("volumes")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to walk up from for roots.toml's [volumes] extra_search_paths.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a table."
)
def data_volumes_cmd(config_dir: Path | None, as_json: bool):
    """List every stamped volume currently attached to this machine.

    Scans platform mount roots (/Volumes on macOS; /mnt, /media/<user>,
    /run/media/<user> on Linux) plus any [volumes] extra_search_paths
    configured in roots.toml. This reports only what is attached right now --
    a stamped volume that is not currently reachable is invisible here. Use
    `data locate` to name a detached volume by volume_id via the replica
    catalog.
    """
    from .cli.data_cmd import data_list_volumes

    volumes = data_list_volumes(config_dir=config_dir)
    if as_json:
        import json

        click.echo(json.dumps(volumes, sort_keys=True, indent=2))
        return
    if not volumes:
        click.echo("No stamped volumes currently attached.")
        return
    click.echo("VOLUME_ID                        LABEL                KIND      MOUNT_PATH")
    for entry in volumes:
        click.echo(
            f"{entry['volume_id']:<32}  {entry['label']:<20} {entry['kind']:<9} {entry['mount_path']}"
        )


@data_group.command("scan")
@click.argument("mounts", nargs=-1, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to walk up from for roots.toml's [volumes] extra_search_paths.",
)
@click.option(
    "--catalog-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the replica catalog file (default: next to roots.toml).",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a summary."
)
def data_scan_cmd(
    mounts: tuple[Path, ...], config_dir: Path | None, catalog_path: Path | None, as_json: bool
):
    """Index runs found on MOUNTS into the replica catalog.

    Scans every currently attached stamped volume when no MOUNTS are given.
    Each mount must already be stamped (`data init-volume`). Walks for
    published input manifests (`raw_outputs/input_manifest/`), registering
    one replica per run root found, keyed by that run's dataset digest.
    """
    from .cli.data_cmd import data_scan

    try:
        result = data_scan(
            [str(mount) for mount in mounts] or None,
            config_dir=config_dir,
            catalog_path=catalog_path,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        import json

        click.echo(json.dumps(result, sort_keys=True, indent=2))
        return
    scanned = result["scanned"]
    if not scanned:
        click.echo("No volumes to scan (none attached, and none named).")
        return
    for entry in scanned:
        click.echo(
            f"{entry['mount']}: {len(entry['runs'])} raw dataset(s), "
            f"{len(entry['analysis_locations'])} analysis location(s)"
        )
        for run in entry["runs"]:
            if run["warning"]:
                click.echo(f"  raw {run['path']}: WARNING: {run['warning']}")
            else:
                click.echo(f"  raw {run['path']}: {run['digest']}")
        for location in entry["analysis_locations"]:
            if location["warning"]:
                click.echo(f"  analysis {location['path']}: WARNING: {location['warning']}")
            else:
                click.echo(f"  analysis {location['path']}: {location['experiment_uid']}")


@data_group.command("locate")
@click.argument("target")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to walk up from for roots.toml's [volumes] extra_search_paths.",
)
@click.option(
    "--catalog-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the replica catalog file (default: next to roots.toml).",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a summary."
)
def data_locate_cmd(target: str, config_dir: Path | None, catalog_path: Path | None, as_json: bool):
    """Show every catalogued replica of TARGET's dataset, and which are attached.

    TARGET is a run root directory, a resolved_input_manifest.json path, or a
    bare sha256 dataset digest. Answers while every replica's volume is
    unplugged -- that is the point of a catalog.
    """
    from .cli.data_cmd import data_locate

    try:
        result = data_locate(target, config_dir=config_dir, catalog_path=catalog_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        import json

        click.echo(json.dumps(result, sort_keys=True, indent=2))
        return
    click.echo(f"dataset {result['dataset_digest']}")
    if not result["replicas"]:
        click.echo("  no catalogued replicas")
        return
    for replica in result["replicas"]:
        status = "attached" if replica["attached"] else "not attached"
        where = replica["resolved_path"] or f"{replica['volume_id']}:{replica['path']}"
        click.echo(f"  [{status}] {where} (verified {replica['verified_at']})")


@data_group.command("verify")
@click.argument("target")
@click.option(
    "--volume", "volume_id", default=None, help="Verify only the replica on this volume_id."
)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to walk up from for roots.toml's [volumes] extra_search_paths.",
)
@click.option(
    "--catalog-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the replica catalog file (default: next to roots.toml).",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a summary."
)
def data_verify_cmd(
    target: str,
    volume_id: str | None,
    config_dir: Path | None,
    catalog_path: Path | None,
    as_json: bool,
):
    """Re-checksum TARGET's declared raw sources against every attached replica.

    TARGET is a run root directory, a resolved_input_manifest.json path, or a
    bare sha256 dataset digest. Only declared sources currently reachable on
    disk are checked; an archived, offline source is reported separately
    rather than as a failure. Exits non-zero if any reachable source's
    checksum no longer matches what its manifest recorded.
    """
    from .cli.data_cmd import data_verify

    try:
        result = data_verify(
            target, volume_id=volume_id, config_dir=config_dir, catalog_path=catalog_path
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        import json

        click.echo(json.dumps(result, sort_keys=True, indent=2))
    else:
        click.echo(f"dataset {result['dataset_digest']}")
        if not result["results"]:
            click.echo("  no catalogued replicas to verify")
        for entry in result["results"]:
            if entry["status"] == "not_attached":
                click.echo(f"  {entry['volume_id']}: not attached")
            elif entry["status"] == "manifest_unreadable":
                click.echo(f"  {entry['volume_id']}: manifest unreadable: {entry['detail']}")
            else:
                click.echo(
                    f"  {entry['volume_id']}: {entry['status']} "
                    f"({entry['checked']} checked, {entry['mismatches']} mismatch(es), "
                    f"{entry['unreachable']} unreachable)"
                )
                for row in entry["rows"]:
                    click.echo(f"    {row['status']}: {row['path']}")

    if any(entry.get("status") == "mismatch" for entry in result["results"]):
        raise click.exceptions.Exit(1)


@data_group.command("localize")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--apply",
    "do_apply",
    is_flag=True,
    help="Copy the files and write a new, localized config. Default is a dry run.",
)
@click.option(
    "--out",
    "out_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the localized config (default: <config>.localized<suffix>).",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a summary."
)
def data_localize_cmd(
    config_path: Path, do_apply: bool, out_config_path: Path | None, as_json: bool
):
    """Copy CONFIG_PATH's small referenced inputs into its own output directory.

    Copies fasta, the BED region files, the sample sheet, and any barcode/UMI
    YAML -- never the raw input itself. Makes the analyses tree
    self-contained: no named root, volume stamp, or replica catalog required
    to read it elsewhere. Defaults to a dry run reporting what would be
    copied and its total size; --apply copies the files and writes a new
    config pointing at the copies. The original config is never modified.
    """
    from .cli.data_cmd import data_localize

    try:
        result = data_localize(config_path, apply=do_apply, out_config_path=out_config_path)
    except (ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        import json

        click.echo(json.dumps(result, sort_keys=True, indent=2))
        return
    if not result["items"]:
        click.echo("Nothing to localize -- no fasta/BED/sample-sheet/barcode-UMI fields declared.")
        return
    for item in result["items"]:
        click.echo(f"  {item['field']}: {item['source']} ({item['size_bytes']} bytes)")
    click.echo(f"Total: {result['total_bytes']} bytes across {len(result['items'])} file(s)")
    if result["applied"]:
        click.echo(
            f"Copied {len(result['copied_fields'])} file(s); wrote {result['new_config_path']}"
        )
    else:
        click.echo("Dry run -- pass --apply to copy these files and write a localized config.")


@data_group.command("init")
@click.argument("lab_root", type=click.Path(path_type=Path))
@click.option(
    "--stamp-volume",
    is_flag=True,
    help="Also stamp the volume LAB_ROOT is on (see data init-volume).",
)
@click.option(
    "--label",
    default=None,
    help="Label for the stamp, with --stamp-volume (default: LAB_ROOT's name).",
)
@click.option(
    "--kind",
    type=click.Choice(["working", "archive", "backup"], case_sensitive=False),
    default="working",
    show_default=True,
    help="Kind for the stamp, with --stamp-volume.",
)
def data_init_cmd(lab_root: Path, stamp_volume: bool, label: str | None, kind: str):
    """Scaffold a new lab directory tree at LAB_ROOT: data/ + analyses/{runs,projects}/.

    Mirrors `project init`, one level up: LAB_ROOT holds immutable raw
    instrument output under data/ and regenerable pipeline output under
    analyses/, per the directory organization tutorial. Idempotent --
    re-running only fills in whatever is still missing, and never touches
    data already collected under data/.

    --stamp-volume also gives LAB_ROOT a permanent volume identity
    (PSR-08), so it can be found on any machine it is later attached to
    regardless of mount point or name -- only discoverable that way if
    LAB_ROOT is itself the volume's own mount point, not a subdirectory of
    a larger drive.
    """
    from .cli.data_cmd import data_init

    created, stamp_result = data_init(
        lab_root, stamp_volume=stamp_volume, label=label, kind=kind.lower()
    )
    if created:
        click.echo(f"Created: {', '.join(created)}")
    else:
        click.echo(f"{lab_root} already scaffolded; nothing to create.")

    if stamp_result is not None:
        stamp_dict, was_created = stamp_result
        verb = "Stamped" if was_created else "Already stamped"
        click.echo(
            f"{verb} volume for {lab_root}: {stamp_dict['volume_id']} "
            f"({stamp_dict['label']}, {stamp_dict['kind']})"
        )
    else:
        click.echo(
            "Not stamped. Pass --stamp-volume, or run 'data init-volume' separately, to "
            "give this drive a portable identity."
        )


@data_group.command("status")
@click.argument("targets", nargs=-1)
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to walk up from for roots.toml's [volumes] extra_search_paths.",
)
@click.option(
    "--catalog-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the replica catalog file (default: next to roots.toml).",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a summary."
)
def data_status_cmd(
    targets: tuple[str, ...], config_dir: Path | None, catalog_path: Path | None, as_json: bool
):
    """Show where every known run's data and analyses are, and their locality.

    Each TARGETS entry is a run root directory or a bare experiment_uid.
    Omitted, every run in the analysis-location catalog (built by `data
    scan`) is reported. For each run: every catalogued analysis location and
    whether it's attached; pairwise ahead/behind/diverged/pointer_conflict
    locality between attached locations (`PSR-17`); and, when at least one
    location is attached and reachable, its raw dataset's catalogued
    replicas and which are attached.
    """
    from .cli.data_cmd import data_status

    try:
        result = data_status(
            list(targets) if targets else None, config_dir=config_dir, catalog_path=catalog_path
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        import json

        click.echo(json.dumps(result, sort_keys=True, indent=2))
        return
    if not result["runs"]:
        click.echo("No runs known to the analysis-location catalog. Run 'data scan' first.")
        return
    for run in result["runs"]:
        click.echo(f"run {run['experiment_uid']}")
        for location in run["locations"]:
            status = "attached" if location["attached"] else "not attached"
            where = location.get("resolved_path") or f"{location['volume_id']}:{location['path']}"
            click.echo(f"  analysis [{status}] {where}")
        for comparison in run["comparisons"]:
            for stage in comparison["stages"]:
                if stage["state"] != "identical":
                    click.echo(
                        f"    {stage['kind']}: {stage['state']} "
                        f"({comparison['a']} vs {comparison['b']})"
                    )
        if run["raw"] is not None:
            click.echo(f"  raw dataset {run['raw']['digest']}")
            for replica in run["raw"]["replicas"]:
                status = "attached" if replica["attached"] else "not attached"
                click.echo(f"    raw [{status}] {replica['volume_id']}:{replica['path']}")


@data_group.command("sync")
@click.argument("target")
@click.option("--from", "from_volume", default=None, help="Source volume_id (with --to).")
@click.option("--to", "to_volume", default=None, help="Destination volume_id (with --from).")
@click.option("--dry-run", is_flag=True, help="Classify and report without copying anything.")
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to walk up from for roots.toml's [volumes] extra_search_paths.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON instead of a summary."
)
def data_sync_cmd(
    target: str,
    from_volume: str | None,
    to_volume: str | None,
    dry_run: bool,
    config_dir: Path | None,
    as_json: bool,
):
    """Additively sync TARGET's generations between two attached analysis locations.

    TARGET is a run root directory or a bare experiment_uid. With no
    --from/--to, exactly two of the run's catalogued locations (`data scan`)
    must currently be attached. For each stage: `ahead`/`behind` copies the
    missing generations across (immutable and content-addressed, so this can
    never corrupt anything and is safe to re-run); `identical` does nothing;
    `diverged`/`pointer_conflict` are reported and never resolved -- sync
    never picks a side, and never moves a current.json pointer.
    """
    from .cli.data_cmd import data_sync

    try:
        result = data_sync(
            target,
            from_volume=from_volume,
            to_volume=to_volume,
            dry_run=dry_run,
            config_dir=config_dir,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        import json

        click.echo(json.dumps(result, sort_keys=True, indent=2))
    else:
        click.echo(
            f"run {result['experiment_uid']}: "
            f"{result['location_a']['volume_id']} <-> {result['location_b']['volume_id']}"
            f"{' (dry run)' if dry_run else ''}"
        )
        for stage in result["stages"]:
            if stage["skipped_reason"]:
                click.echo(f"  {stage['kind']}: {stage['state']} -- {stage['skipped_reason']}")
            elif stage["copied_a_to_b"] or stage["copied_b_to_a"]:
                click.echo(
                    f"  {stage['kind']}: copied {list(stage['copied_a_to_b'])} a->b, "
                    f"{list(stage['copied_b_to_a'])} b->a"
                )
            else:
                click.echo(f"  {stage['kind']}: {stage['state']}")

    if any(stage["skipped_reason"] for stage in result["stages"]):
        raise click.exceptions.Exit(1)


##########################################

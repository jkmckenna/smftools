from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from ._version import __version__
from .cli.recipes import full_flow
from .logging_utils import get_logger, setup_logging
from .memory_guard import enable_aggregate_memory_cap


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


####### Load anndata from raw data ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def raw(config_path):
    """Prepare BAM artifacts and write the ragged raw store."""
    from .cli.raw_adata import raw_adata

    raw_adata(config_path)


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
    from .cli.preprocess_adata import preprocess_adata

    preprocess_adata(config_path)


##########################################


####### Spatial ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def spatial(config_path):
    """Spatial signal analysis"""
    from .cli.spatial_adata import spatial_adata

    spatial_adata(config_path)


##########################################


####### HMM ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def hmm(config_path):
    """HMM feature annotations and plotting"""
    from .cli.hmm_adata import hmm_adata

    hmm_adata(config_path)


##########################################


####### Latent ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def latent(config_path):
    """Latent representations of signal"""
    from .cli.latent_adata import latent_adata

    latent_adata(config_path)


##########################################


####### Variant ###########
@experiment_group.command()
@click.argument("config_path", type=click.Path(exists=True))
def variant(config_path):
    """Sequence variation analyses"""
    from .cli.variant_adata import variant_adata

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
    """Workflow: raw preprocess spatial hmm."""
    full_flow(config_path)


##########################################


####### batch command ###########
@experiment_group.command()
@click.argument(
    "task",
    type=click.Choice(
        ["raw", "load", "preprocess", "spatial", "variant", "hmm"],
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
def batch(task, config_table: Path, column: str, sep: str | None):
    """
    Run a TASK (raw, load, preprocess, spatial, variant, hmm) on multiple CONFIG_PATHs
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
        from .cli.raw_adata import raw_adata

        return raw_adata(cfg_path)

    def _load(cfg_path: str):
        from .cli.load_adata import load_dense_cache

        return load_dense_cache(cfg_path)

    def _preprocess(cfg_path: str):
        from .cli.preprocess_adata import preprocess_adata

        return preprocess_adata(cfg_path)

    def _spatial(cfg_path: str):
        from .cli.spatial_adata import spatial_adata

        return spatial_adata(cfg_path)

    def _variant(cfg_path: str):
        from .cli.variant_adata import variant_adata

        return variant_adata(cfg_path)

    def _hmm(cfg_path: str):
        from .cli.hmm_adata import hmm_adata

        return hmm_adata(cfg_path)

    task_funcs = {
        "raw": _raw,
        "load": _load,
        "preprocess": _preprocess,
        "spatial": _spatial,
        "variant": _variant,
        "hmm": _hmm,
    }

    func = task_funcs[task]

    click.echo(f"Running task '{task}' on {len(config_paths)} config paths from {config_table}")

    # ----------------------------
    # Loop over paths
    # ----------------------------
    for i, cfg in enumerate(config_paths, start=1):
        if not cfg.exists():
            click.echo(f"[{i}/{len(config_paths)}] SKIP (missing): {cfg}")
            continue

        click.echo(f"[{i}/{len(config_paths)}] {task} → {cfg}")

        try:
            func(str(cfg))  # underlying functions take a string path
        except Exception as e:
            click.echo(f"  ERROR on {cfg}: {e}")

    click.echo("Batch processing complete.")


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
def project_init_cmd(project_dir: Path):
    """Initialize a project directory + registry."""
    from .cli.project_cmd import project_init

    path = project_init(project_dir)
    click.echo(f"Initialized project registry: {path}")


@project_group.command("add")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("experiment_dir", type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.option("--id", "experiment_id", default=None, help="Explicit experiment id.")
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

    exp_id, entry, conflicts = project_add(
        project_dir, experiment_dir, experiment_id=experiment_id, name=name, stage=stage
    )
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


@project_group.command("materialize")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("canonical_reference")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output .h5ad(.gz).")
@click.option("--set", "set_name", default=None, help="Restrict to a named experiment set.")
@click.option("--modality", default=None, help="Restrict to a modality.")
@click.option(
    "--stage",
    default=None,
    help=(
        "Pipeline stage to materialize per experiment (raw/preprocess/spatial/hmm/"
        "latent/variant/chimeric). Default: most-derived stage available per "
        "experiment, since a later stage already carries forward earlier stages' data."
    ),
)
@click.option("--start", type=int, default=None, help="Genomic window start (with --end).")
@click.option("--end", type=int, default=None, help="Genomic window end (with --start).")
@click.option(
    "--read-metrics",
    is_flag=True,
    help="Also attach spatial-stage per-read outputs (autocorrelation, Lomb-Scargle) when available.",
)
def project_materialize_cmd(
    project_dir, canonical_reference, output, set_name, modality, stage, start, end, read_metrics
):
    """Materialize CANONICAL_REFERENCE across matching experiments into one AnnData."""
    from .cli.project_cmd import project_materialize

    out = project_materialize(
        project_dir,
        canonical_reference,
        output,
        set_name=set_name,
        modality=modality,
        stage=stage,
        start=start,
        end=end,
        read_metrics=read_metrics,
    )
    click.echo(f"Wrote {out}")


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
        [item.strip() for item in experiments.split(",") if item.strip()]
        if experiments
        else None
    )
    out = export_fastq_for_project(
        project_dir,
        outdir,
        experiments=experiment_list,
        allow_unfiltered=allow_unfiltered,
        gzip_output=not no_gzip,
    )
    click.echo(f"Wrote FASTQ export to: {out}")


##########################################


####### Plot current traces ###########
@experiment_group.command("plot-current")
@click.argument("config_path", type=click.Path(exists=True))
def plot_current(config_path):
    """Plot nanopore current traces for specified reads."""
    from .cli.plot_current import plot_current as plot_current_fn

    plot_current_fn(config_path)


##########################################

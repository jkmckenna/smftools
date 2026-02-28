from __future__ import annotations

import logging
from pathlib import Path
import click
import pandas as pd

from .cli.chimeric_adata import chimeric_adata
from .cli.hmm_adata import hmm_adata
from .cli.latent_adata import latent_adata
from .cli.load_adata import load_adata
from .cli.plot_current import plot_current as plot_current_fn
from .cli.preprocess_adata import preprocess_adata
from .cli.recipes import full_flow
from .cli.spatial_adata import spatial_adata
from .cli.variant_adata import variant_adata
from ._version import __version__
from .informatics.pod5_functions import subsample_pod5
from .logging_utils import get_logger, setup_logging
from .readwrite import concatenate_h5ads


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
    _configure_multiprocessing()


####### Load anndata from raw data ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def load(config_path):
    """Load raw data into AnnData."""
    load_adata(config_path)


##########################################


####### Preprocessing ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def preprocess(config_path):
    """Preprocessing."""
    preprocess_adata(config_path)


##########################################


####### Spatial ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def spatial(config_path):
    """Spatial signal analysis"""
    spatial_adata(config_path)


##########################################


####### HMM ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def hmm(config_path):
    """HMM feature annotations and plotting"""
    hmm_adata(config_path)


##########################################


####### Latent ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def latent(config_path):
    """Latent representations of signal"""
    latent_adata(config_path)


##########################################


####### Variant ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def variant(config_path):
    """Sequence variation analyses"""
    variant_adata(config_path)


##########################################


####### Chimeric ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def chimeric(config_path):
    """Finding putative PCR chimeras"""
    chimeric_adata(config_path)


##########################################


####### Recipes ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def full(config_path):
    """Workflow: load preprocess spatial variant chimeric hmm latent"""
    full_flow(config_path)


##########################################


####### batch command ###########
@cli.command()
@click.argument(
    "task",
    type=click.Choice(["load", "preprocess", "spatial", "variant", "hmm"], case_sensitive=False),
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
    Run a TASK (load, preprocess, spatial, variant, hmm) on multiple CONFIG_PATHs
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
    task_funcs = {
        "load": load_adata,
        "preprocess": preprocess_adata,
        "spatial": spatial_adata,
        "variant": variant_adata,
        "hmm": hmm_adata,
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
@cli.command("concatenate")
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

        smftools concatenate experiment_config.csv

        smftools concatenate experiment_config.csv --recompute-pp-vars

        smftools concatenate experiment_config.csv --input-dir ./variant_h5ads/
    """
    from .cli.helpers import load_experiment_config

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


####### Plot current traces ###########
@cli.command("plot-current")
@click.argument("config_path", type=click.Path(exists=True))
def plot_current(config_path):
    """Plot nanopore current traces for specified reads."""
    plot_current_fn(config_path)


##########################################

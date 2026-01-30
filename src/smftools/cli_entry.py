from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import click
import pandas as pd

from .cli.hmm_adata import hmm_adata
from .cli.latent_adata import latent_adata
from .cli.load_adata import load_adata
from .cli.preprocess_adata import preprocess_adata
from .cli.spatial_adata import spatial_adata
from .cli.variant_adata import variant_adata
from .cli.chimeric_adata import chimeric_adata

from .cli.recipes import full_flow

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
    type=click.Choice(["load", "preprocess", "spatial", "hmm"], case_sensitive=False),
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
    Run a TASK (load, preprocess, spatial, hmm) on multiple CONFIG_PATHs
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
@click.argument(
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option(
    "--input-dir",
    "-d",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory containing .h5ad/.h5ad.gz files to concatenate.",
)
@click.option(
    "--csv-path",
    "-c",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="CSV/TSV/TXT containing file paths of h5ad files.",
)
@click.option(
    "--csv-column",
    "-C",
    default="h5ad_path",
    help="Column in the CSV containing file paths (ignored for TXT).",
    show_default=True,
)
@click.option(
    "--suffix",
    "-s",
    multiple=True,
    default=[".h5ad", ".h5ad.gz"],
    help="Allowed file suffixes (repeatable).",
    show_default=True,
)
@click.option(
    "--delete",
    is_flag=False,
    help="Delete input .h5ad files after concatenation.",
)
@click.option(
    "--restore",
    is_flag=True,
    help="Restore .h5ad backups during reading.",
)
def concatenate_cmd(
    output_path: Path,
    input_dir: Path | None,
    csv_path: Path | None,
    csv_column: str,
    suffix: Sequence[str],
    delete: bool,
    restore: bool,
):
    """
    Concatenate multiple .h5ad files into a single output file.

    Two modes:

        smftools concatenate out.h5ad.gz --input-dir ./dir

        smftools concatenate out.h5ad.gz --csv-path paths.csv --csv-column h5ad_path

    TXT input also works (one file path per line).

    Uses safe_read_h5ad() and safe_write_h5ad().
    """

    if input_dir and csv_path:
        raise click.ClickException("Provide only ONE of --input-dir or --csv-path.")

    try:
        out = concatenate_h5ads(
            output_path=output_path,
            input_dir=input_dir,
            csv_path=csv_path,
            csv_column=csv_column,
            file_suffixes=tuple(suffix),
            delete_inputs=delete,
            restore_backups=restore,
        )
        click.echo(f"Concatenated file written to: {out}")

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

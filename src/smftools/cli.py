import click
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from . import load_adata
from .readwrite import merge_barcoded_anndatas_core, safe_read_h5ad, safe_write_h5ad

@click.group()
def cli():
    """Command-line interface for smftools."""
    pass

####### Main processing workflow ###########
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def load(config_path):
    """Load and process data from CONFIG_PATH."""
    load_adata(config_path)
##########################################


####### Merging existing anndatas from an experiment that used two different demultiplexing rules #######
REQUIRED_KEYS = ("adata_single_path", "adata_double_path")
OPTIONAL_KEYS = (
    "adata_single_backups_path",
    "adata_double_backups_path",
    "output_path",
    "merged_filename",
)

def _read_config_csv(csv_path: Path) -> Dict[str, str]:
    """
    Read a multi-row, two-column CSV of key,value pairs into a dict.

    Supported features:
      - Optional header ("key,value") or none.
      - Comments starting with '#' and blank lines are ignored.
      - If duplicate keys occur, the last one wins.
      - Keys are matched literally against REQUIRED_KEYS/OPTIONAL_KEYS.
    """
    try:
        # Read as two columns regardless of header; comments ignored.
        df = pd.read_csv(
            csv_path,
            dtype=str,
            comment="#",
            header=None,          # treat everything as rows; we'll normalize below
            usecols=[0, 1],
            names=["key", "value"]
        )
    except Exception as e:
        raise click.ClickException(f"Failed to read CSV: {e}") from e

    # Drop completely empty rows
    df = df.fillna("").astype(str)
    df["key"] = df["key"].str.strip()
    df["value"] = df["value"].str.strip()
    df = df[(df["key"] != "") & (df["key"].notna())]

    if df.empty:
        raise click.ClickException("Config CSV is empty after removing comments/blank lines.")

    # Remove an optional header row if present
    if df.iloc[0]["key"].lower() in {"key", "keys"}:
        df = df.iloc[1:]
        df = df[(df["key"] != "") & (df["key"].notna())]
        if df.empty:
            raise click.ClickException("Config CSV contains only a header row.")

    # Build dict; last occurrence of a key wins
    cfg = {}
    for k, v in zip(df["key"], df["value"]):
        cfg[k] = v

    # Validate required keys
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise click.ClickException(
            "Missing required keys in CSV: "
            + ", ".join(missing)
            + "\nExpected keys:\n  - "
            + "\n  - ".join(REQUIRED_KEYS)
            + "\nOptional keys:\n  - "
            + "\n  - ".join(OPTIONAL_KEYS)
        )

    return cfg

def _resolve_output_path(cfg: Dict[str, str], single_path: Path, double_path: Path) -> Path:
    """Decide on the output .h5ad path based on CSV; create directories if needed."""
    merged_filename = cfg.get("merged_filename") or f"merged_{single_path.stem}__{double_path.stem}.h5ad"
    if not merged_filename.endswith(".h5ad"):
        merged_filename += ".h5ad"

    output_path_raw = cfg.get("output_path", "").strip()

    if not output_path_raw:
        out_dir = Path.cwd() / "merged_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / merged_filename

    output_path = Path(output_path_raw)

    if output_path.suffix.lower() == ".h5ad":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    # Treat as directory
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / merged_filename

def _maybe_read_adata(label: str, primary: Path, backups: Optional[Path]):

    if backups:
        click.echo(f"Loading {label} from {primary} with backups at {backups} ...")
        return safe_read_h5ad(primary, backups_path=backups, restore_backups=True)
    else:
        click.echo(f"Loading {label} from {primary} with backups disabled ...")
        return safe_read_h5ad(primary, restore_backups=False)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path))
def merge_barcoded_anndatas(config_path: Path):
    """
    Merge two AnnData objects from the same experiment that were demultiplexed
    under different end-barcoding requirements, using a 1-row CSV for config.

    CSV must include:
      - adata_single_path
      - adata_double_path

    Optional columns:
      - adata_single_backups_path
      - adata_double_backups_path
      - output_path            (file or directory; default: ./merged_output/)
      - merged_filename        (default: merged_<single>__<double>.h5ad)

    Example CSV:

        adata_single_path,adata_double_path,adata_single_backups_path,adata_double_backups_path,output_path,merged_filename
        /path/single.h5ad,/path/double.h5ad,,,,merged_output,merged_run.h5ad
    """
    try:
        cfg = _read_config_csv(config_path)

        single_path = Path(cfg["adata_single_path"]).expanduser().resolve()
        double_path = Path(cfg["adata_double_path"]).expanduser().resolve()

        for p, label in [(single_path, "adata_single_path"), (double_path, "adata_double_path")]:
            if not p.exists():
                raise click.ClickException(f"{label} does not exist: {p}")

        single_backups = Path(cfg["adata_single_backups_path"]).expanduser().resolve() if cfg.get("adata_single_backups_path") else None
        double_backups = Path(cfg["adata_double_backups_path"]).expanduser().resolve() if cfg.get("adata_double_backups_path") else None

        if single_backups and not single_backups.exists():
            raise click.ClickException(f"adata_single_backups_path does not exist: {single_backups}")
        if double_backups and not double_backups.exists():
            raise click.ClickException(f"adata_double_backups_path does not exist: {double_backups}")

        output_path = _resolve_output_path(cfg, single_path, double_path)

        # Load
        adata_single, read_report_single = _maybe_read_adata("single-barcoded AnnData", single_path, single_backups)
        adata_double, read_report_double = _maybe_read_adata("double-barcoded AnnData", double_path, double_backups)

        click.echo("Merging AnnDatas ...")
        merged = merge_barcoded_anndatas_core(adata_single, adata_double)

        click.echo(f"Writing merged AnnData to: {output_path}")
        backup_dir = output_path.cwd() / "merged_backups"
        safe_write_h5ad(merged, output_path, backup=True, backup_dir=backup_dir)

        click.secho(f"Done. Merged AnnData saved to {output_path}", fg="green")

    except click.ClickException:
        raise
    except Exception as e:
        # Surface unexpected errors cleanly
        raise click.ClickException(f"Unexpected error: {e}") from e

################################################################################################################
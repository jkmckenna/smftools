"""CLI wrapper for plotting nanopore current traces from POD5 files."""

from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger
from smftools.readwrite import safe_read_h5ad

from .helpers import get_adata_paths, load_experiment_config, resolve_adata_stage

logger = get_logger(__name__)


def plot_current(config_path: str) -> None:
    """Load config, resolve AnnData + POD5 dir, and plot current traces."""
    from smftools.plotting.pod5_plotting import plot_read_current_traces

    cfg = load_experiment_config(config_path)
    paths = get_adata_paths(cfg)

    # Resolve which AnnData to load
    adata_path, stage = resolve_adata_stage(cfg, paths)
    if adata_path is None:
        raise FileNotFoundError(
            "No AnnData file found. Run a pipeline stage first or set from_adata_stage."
        )

    logger.info(f"Loading AnnData from stage '{stage}': {adata_path}")
    adata, _ = safe_read_h5ad(adata_path)

    # Read IDs to plot
    read_ids = cfg.plot_current_read_ids
    if not read_ids:
        raise ValueError(
            "No read IDs specified. Set 'plot_current_read_ids' in your experiment config."
        )

    # POD5 directory from input_data_path
    pod5_dir = cfg.input_data_path

    # Reference coordinates (from config or derived from adata.var)
    ref_start = cfg.plot_current_reference_start
    ref_end = cfg.plot_current_reference_end
    var_start = cfg.plot_current_var_start
    var_end = cfg.plot_current_var_end

    # Output directory
    output_dir = Path(cfg.output_directory) / "plot_current"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{cfg.experiment_name}_current_traces.png"

    logger.info(f"Plotting current traces for {len(read_ids)} read(s)")
    plot_read_current_traces(
        adata,
        read_ids,
        reference_start=ref_start,
        reference_end=ref_end,
        var_start=var_start,
        var_end=var_end,
        pod5_dir=pod5_dir,
        save_path=save_path,
    )
    logger.info(f"Done. Plot saved to {save_path}")

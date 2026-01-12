from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def load_sample_sheet(
    adata: "ad.AnnData",
    sample_sheet_path: str | Path,
    mapping_key_column: str = "obs_names",
    as_category: bool = True,
    uns_flag: str = "load_sample_sheet_performed",
    force_reload: bool = True,
) -> "ad.AnnData":
    """Load a sample sheet CSV and map metadata into ``adata.obs``.

    Args:
        adata: AnnData object to append sample information to.
        sample_sheet_path: Path to the CSV file.
        mapping_key_column: Column name to map against ``adata.obs_names`` or an obs column.
        as_category: Whether to cast added columns as pandas Categoricals.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_reload: Whether to reload even if ``uns_flag`` is set.

    Returns:
        anndata.AnnData: Updated AnnData object.
    """
    import pandas as pd

    # Only run if not already performed
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_reload:
        # QC already performed; nothing to do
        return

    logger.info("Loading sample sheet...")
    df = pd.read_csv(sample_sheet_path)
    df[mapping_key_column] = df[mapping_key_column].astype(str)

    # If matching against obs_names directly
    if mapping_key_column == "obs_names":
        key_series = adata.obs_names.astype(str)
    else:
        key_series = adata.obs[mapping_key_column].astype(str)

    value_columns = [col for col in df.columns if col != mapping_key_column]

    logger.info("Appending metadata columns: %s", value_columns)
    df = df.set_index(mapping_key_column)

    for col in value_columns:
        mapped = key_series.map(df[col])
        if as_category:
            mapped = mapped.astype("category")
        adata.obs[col] = mapped

    # mark as done
    adata.uns[uns_flag] = True

    if as_category:
        logger.info("Sample sheet metadata successfully added as categories.")
    else:
        logger.info("Metadata added.")
    return adata

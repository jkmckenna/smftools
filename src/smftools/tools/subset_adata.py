# subset_adata

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import anndata as ad


def subset_adata(adata: "ad.AnnData", columns: Sequence[str], cat_type: str = "obs") -> None:
    """Add subset metadata based on categorical values in ``.obs`` or ``.var`` columns.

    Args:
        adata: AnnData object to annotate.
        columns: Obs or var column names to subset by (order matters).
        cat_type: ``"obs"`` or ``"var"``.
    """

    subgroup_name = "_".join(columns)
    if "obs" in cat_type:
        df = adata.obs[columns]
        adata.obs[subgroup_name] = df.apply(lambda row: "_".join(row.astype(str)), axis=1)
        adata.obs[subgroup_name] = adata.obs[subgroup_name].astype("category")
    elif "var" in cat_type:
        df = adata.var[columns]
        adata.var[subgroup_name] = df.apply(lambda row: "_".join(row.astype(str)), axis=1)
        adata.var[subgroup_name] = adata.var[subgroup_name].astype("category")

    return None

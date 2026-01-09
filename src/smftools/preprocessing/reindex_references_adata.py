from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def reindex_references_adata(
    adata,
    reference_col: str = "Reference_strand",
    offsets: dict | None = None,
    new_col: str = "reindexed",
    uns_flag: str = "reindex_references_adata_performed",
    force_redo: bool = False,
):
    """
    Reindex genomic coordinates by adding per-reference offsets.

    BEHAVIOR:
      - If offsets is None or empty, or a reference is missing from offsets,
        create a reindex column for that reference that simply copies var_names.
    """

    import numpy as np

    # ============================================================
    # 1. Skip if already done
    # ============================================================
    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        logger.info("%s already set; skipping. Use force_redo=True to recompute.", uns_flag)
        return None

    # Normalize offsets
    if offsets is None:
        offsets = {}
    elif not isinstance(offsets, dict):
        raise TypeError("offsets must be a dict {ref: int} or None.")

    # ============================================================
    # 2. Ensure var_names are numeric
    # ============================================================
    try:
        var_coords = adata.var_names.astype(int)
    except Exception as e:
        raise ValueError(
            "reindex_references_adata requires adata.var_names to be integer-like."
        ) from e

    # ============================================================
    # 3. Gather all references
    # ============================================================
    ref_series = adata.obs[reference_col]
    references = ref_series.cat.categories if hasattr(ref_series, "cat") else ref_series.unique()

    # ============================================================
    # 4. Create reindexed columns
    # ============================================================
    for ref in references:
        colname = f"{ref}_{new_col}"

        # Case 1: No offset provided → identity mapping
        if ref not in offsets:
            logger.info("No offset for ref=%r; using identity positions.", ref)
            adata.var[colname] = var_coords
            continue

        offset_value = offsets[ref]

        # Case 2: offset explicitly None → identity mapping
        if offset_value is None:
            logger.info("Offset for ref=%r is None; using identity positions.", ref)
            adata.var[colname] = var_coords
            continue

        # Case 3: real shift
        if not isinstance(offset_value, (int, np.integer)):
            raise TypeError(
                f"Offset for reference {ref!r} must be an integer or None. Got {offset_value!r}"
            )

        adata.var[colname] = var_coords + offset_value
        logger.info("Added reindexed column '%s' (offset=%s).", colname, offset_value)

    # ============================================================
    # 5. Mark complete
    # ============================================================
    adata.uns[uns_flag] = True
    logger.info("Reindexing complete!")

    return None

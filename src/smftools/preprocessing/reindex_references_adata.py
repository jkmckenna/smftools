from __future__ import annotations

from typing import TYPE_CHECKING

from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def reindex_references_adata(
    adata: "ad.AnnData",
    reference_col: str = "Reference_strand",
    offsets: dict | None = None,
    new_col: str = "reindexed",
    uns_flag: str = "reindex_references_adata_performed",
    force_redo: bool = False,
    invert: dict | bool | None = None,
) -> None:
    """Reindex genomic coordinates by adding per-reference offsets.

    Args:
        adata: AnnData object.
        reference_col: Obs column containing reference identifiers.
        offsets: Mapping of reference to integer offset.
        new_col: Suffix for generated reindexed columns.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        invert: Per-reference display-inversion flag(s). Either a single bool
            applied to every reference, a ``{ref: bool}`` mapping, or ``None``
            (no inversion, the default). When a reference is inverted, its
            reindexed coordinate's *sign* is flipped (``-(var_coords +
            offset)`` instead of ``var_coords + offset``) so that "left of the
            anchor is negative, right of the anchor is positive" still holds
            after the reference is displayed in reverse column order. This
            never touches ``X``/layers/``var_names`` -- it is purely a
            reinterpretation of the reindexed coordinate value; callers that
            render columns are responsible for reordering them to match (see
            ``plotting.plotting_utils._ordered_columns``).

    Notes:
        If ``offsets`` is ``None`` or missing a reference, the new column mirrors
        the existing ``var_names`` values (subject to the sign flip above).
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

    # Normalize invert: accept a single bool (applied to every reference), a
    # {ref: bool} mapping, or None (no inversion anywhere).
    if invert is None:
        invert_default = False
        invert_map: dict = {}
    elif isinstance(invert, bool):
        invert_default = invert
        invert_map = {}
    elif isinstance(invert, dict):
        invert_default = False
        invert_map = invert
    else:
        raise TypeError("invert must be a dict {ref: bool}, a bool, or None.")

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
        sign = -1 if bool(invert_map.get(ref, invert_default)) else 1

        # Case 1: No offset provided → identity mapping (sign still applies)
        if ref not in offsets:
            logger.info("No offset for ref=%r; using identity positions.", ref)
            adata.var[colname] = sign * var_coords
            continue

        offset_value = offsets[ref]

        # Case 2: offset explicitly None → identity mapping (sign still applies)
        if offset_value is None:
            logger.info("Offset for ref=%r is None; using identity positions.", ref)
            adata.var[colname] = sign * var_coords
            continue

        # Case 3: real shift
        if not isinstance(offset_value, (int, np.integer)):
            raise TypeError(
                f"Offset for reference {ref!r} must be an integer or None. Got {offset_value!r}"
            )

        adata.var[colname] = sign * (var_coords + offset_value)
        logger.info(
            "Added reindexed column '%s' (offset=%s, invert=%s).", colname, offset_value, sign < 0
        )

    # ============================================================
    # 5. Mark complete
    # ============================================================
    adata.uns[uns_flag] = True
    logger.info("Reindexing complete!")

    return None

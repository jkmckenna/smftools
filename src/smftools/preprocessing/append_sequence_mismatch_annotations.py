from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from smftools.logging_utils import get_logger
from smftools.tools.sequence_alignment import align_sequences_with_mismatches

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def _format_mismatch_identity(event: str, seq1_base: str | None, seq2_base: str | None) -> str:
    if event == "substitution":
        return f"{seq1_base}->{seq2_base}"
    if event == "insertion":
        return f"ins:{seq2_base}"
    return f"del:{seq1_base}"


def append_sequence_mismatch_annotations(
    adata: "ad.AnnData",
    seq1_column: str,
    seq2_column: str,
    output_prefix: str | None = None,
    match_score: int = 1,
    mismatch_score: int = -1,
    gap_score: int = -2,
    ignore_n: bool = True,
    bypass: bool = False,
    force_redo: bool = False,
    uns_flag: str = "append_sequence_mismatch_annotations_performed",
) -> None:
    """Append mismatch annotations by aligning full reference sequences.

    Extracts the full reference sequences from per-position base columns in
    ``adata.var``, performs a single global alignment, and maps mismatches
    (substitutions, insertions, deletions) back to ``adata.var`` indices.

    Results stored in ``adata.var``:
      - ``{prefix}_mismatch_type``: Per-position str — ``"substitution"``,
        ``"insertion"``, ``"deletion"``, or ``""`` (no mismatch).
      - ``{prefix}_mismatch_identity``: Per-position str — e.g. ``"A->G"``,
        ``"ins:T"``, ``"del:C"``, or ``""``).
      - ``{prefix}_is_mismatch``: Per-position bool flag.

    Args:
        adata: AnnData object.
        seq1_column: Column in ``adata.var`` with per-position bases for reference 1.
        seq2_column: Column in ``adata.var`` with per-position bases for reference 2.
        output_prefix: Prefix for output columns. Defaults to ``{seq1_column}__{seq2_column}``.
        match_score: Alignment match score.
        mismatch_score: Alignment mismatch score.
        gap_score: Alignment gap score.
        ignore_n: Whether to ignore mismatches involving ``N`` bases.
        bypass: Whether to skip processing.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
    """
    already = bool(adata.uns.get(uns_flag, False))
    if (already and not force_redo) or bypass:
        return

    if seq1_column not in adata.var:
        raise KeyError(f"Sequence column '{seq1_column}' not found in adata.var")
    if seq2_column not in adata.var:
        raise KeyError(f"Sequence column '{seq2_column}' not found in adata.var")

    output_prefix = output_prefix or f"{seq1_column}__{seq2_column}"

    seq1_series = adata.var[seq1_column]
    seq2_series = adata.var[seq2_column]
    n_vars = adata.shape[1]

    # ---- Build full sequences from positions where each ref has a valid (non-N) base ----
    valid1_mask = seq1_series.notna() & (seq1_series != "N")
    valid2_mask = seq2_series.notna() & (seq2_series != "N")

    # var indices (integers) for each valid base
    var_indices_1 = np.where(valid1_mask.values)[0]
    var_indices_2 = np.where(valid2_mask.values)[0]

    full_seq1 = "".join(str(seq1_series.iloc[i]) for i in var_indices_1)
    full_seq2 = "".join(str(seq2_series.iloc[i]) for i in var_indices_2)

    logger.info(
        "Aligning full sequences: '%s' (%d bases) vs '%s' (%d bases).",
        seq1_column,
        len(full_seq1),
        seq2_column,
        len(full_seq2),
    )

    # ---- Global alignment ----
    aligned_seq1, aligned_seq2, mismatches = align_sequences_with_mismatches(
        full_seq1,
        full_seq2,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        ignore_n=ignore_n,
    )

    logger.info(
        "Alignment complete. Aligned length: %d, mismatches: %d.",
        len(aligned_seq1),
        len(mismatches),
    )

    # ---- Map alignment mismatches back to var indices ----
    mismatch_type_arr = [""] * n_vars
    mismatch_identity_arr = [""] * n_vars
    is_mismatch_arr = np.zeros(n_vars, dtype=bool)

    # For substitutions, store the paired var indices from both references.
    # This is needed because indels shift the coordinate systems so that the
    # same alignment column maps to different var indices in each reference.
    substitution_map: list[dict] = []

    for mm in mismatches:
        # Determine which var index this mismatch maps to.
        # For substitutions and deletions, seq1_pos is defined.
        # For insertions, only seq2_pos is defined (gap in seq1).
        if mm.seq1_pos is not None:
            var_idx = int(var_indices_1[mm.seq1_pos])
        elif mm.seq2_pos is not None:
            var_idx = int(var_indices_2[mm.seq2_pos])
        else:
            continue

        mismatch_type_arr[var_idx] = mm.event
        mismatch_identity_arr[var_idx] = _format_mismatch_identity(
            mm.event, mm.seq1_base, mm.seq2_base
        )
        is_mismatch_arr[var_idx] = True

        if mm.event == "substitution" and mm.seq1_pos is not None and mm.seq2_pos is not None:
            substitution_map.append(
                {
                    "seq1_var_idx": int(var_indices_1[mm.seq1_pos]),
                    "seq2_var_idx": int(var_indices_2[mm.seq2_pos]),
                    "seq1_base": mm.seq1_base,
                    "seq2_base": mm.seq2_base,
                }
            )

    adata.var[f"{output_prefix}_mismatch_type"] = pd.Series(
        mismatch_type_arr, index=adata.var.index
    )
    adata.var[f"{output_prefix}_mismatch_identity"] = pd.Series(
        mismatch_identity_arr, index=adata.var.index
    )
    adata.var[f"{output_prefix}_is_mismatch"] = pd.Series(is_mismatch_arr, index=adata.var.index)
    # Store substitution map as a DataFrame in adata.uns (h5ad-serializable)
    if substitution_map:
        adata.uns[f"{output_prefix}_substitution_map"] = pd.DataFrame(substitution_map)
    adata.uns[uns_flag] = True

    n_sub = sum(1 for t in mismatch_type_arr if t == "substitution")
    n_ins = sum(1 for t in mismatch_type_arr if t == "insertion")
    n_del = sum(1 for t in mismatch_type_arr if t == "deletion")
    logger.info(
        "Mismatch annotations: %d substitutions, %d insertions, %d deletions.",
        n_sub,
        n_ins,
        n_del,
    )

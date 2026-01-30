from __future__ import annotations

from typing import TYPE_CHECKING

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
    gap_score: int = -1,
    ignore_n: bool = True,
    bypass: bool = False,
    force_redo: bool = False,
    uns_flag: str = "append_sequence_mismatch_annotations_performed",
) -> None:
    """Append mismatch positions, types, and identities for sequence columns.

    Args:
        adata: AnnData object.
        seq1_column: Column in ``adata.var`` containing the first sequence.
        seq2_column: Column in ``adata.var`` containing the second sequence.
        output_prefix: Prefix for mismatch columns. Defaults to ``{seq1_column}__{seq2_column}``.
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
    positions: list[list[tuple[int | None, int | None]]] = []
    types: list[list[str]] = []
    identities: list[list[str]] = []

    for seq1, seq2 in zip(adata.var[seq1_column], adata.var[seq2_column], strict=True):
        if pd.isna(seq1) or pd.isna(seq2):
            positions.append([])
            types.append([])
            identities.append([])
            continue

        _, _, mismatches = align_sequences_with_mismatches(
            str(seq1),
            str(seq2),
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            ignore_n=ignore_n,
        )
        positions.append([(mismatch.seq1_pos, mismatch.seq2_pos) for mismatch in mismatches])
        types.append([mismatch.event for mismatch in mismatches])
        identities.append(
            [
                _format_mismatch_identity(mismatch.event, mismatch.seq1_base, mismatch.seq2_base)
                for mismatch in mismatches
            ]
        )

    adata.var[f"{output_prefix}_mismatch_positions"] = positions
    adata.var[f"{output_prefix}_mismatch_types"] = types
    adata.var[f"{output_prefix}_mismatch_identities"] = identities
    adata.uns[uns_flag] = True

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.logging_utils import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


def append_mismatch_frequency_sites(
    adata: "ad.AnnData",
    ref_column: str = "Reference_strand",
    mismatch_layer: str = "mismatch_integer_encoding",
    read_span_layer: str = "read_span_mask",
    mismatch_frequency_range: Sequence[float] | None = (0.05, 0.95),
    uns_flag: str = "append_mismatch_frequency_sites_performed",
    force_redo: bool = False,
    bypass: bool = False,
) -> None:
    """Append mismatch frequency metadata and variable-site flags per reference.

    Args:
        adata: AnnData object.
        ref_column: Obs column defining reference categories.
        mismatch_layer: Layer containing mismatch integer encodings.
        read_span_layer: Layer containing read span masks (1=covered, 0=not covered).
        mismatch_frequency_range: Lower/upper bounds (inclusive) for variable site flagging.
        uns_flag: Flag in ``adata.uns`` indicating prior completion.
        force_redo: Whether to rerun even if ``uns_flag`` is set.
        bypass: Whether to skip running this step.
    """
    if bypass:
        return

    already = bool(adata.uns.get(uns_flag, False))
    if already and not force_redo:
        return

    if mismatch_layer not in adata.layers:
        logger.debug(
            "Mismatch layer '%s' not found; skipping mismatch frequency step.", mismatch_layer
        )
        return

    mismatch_map = adata.uns.get("mismatch_integer_encoding_map", {})
    if not mismatch_map:
        logger.debug("Mismatch encoding map not found; skipping mismatch frequency step.")
        return

    n_value = mismatch_map.get("N", MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"])
    pad_value = mismatch_map.get("PAD", MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"])

    base_int_to_label = {
        int(value): str(base)
        for base, value in mismatch_map.items()
        if base not in {"N", "PAD"} and isinstance(value, (int, np.integer))
    }
    if not base_int_to_label:
        logger.debug("Mismatch encoding map missing base labels; skipping mismatch frequency step.")
        return

    has_span_mask = read_span_layer in adata.layers
    if not has_span_mask:
        logger.debug(
            "Read span mask '%s' not found; mismatch frequencies will be computed over all reads.",
            read_span_layer,
        )

    references = adata.obs[ref_column].cat.categories
    n_vars = adata.shape[1]

    if mismatch_frequency_range is None:
        mismatch_frequency_range = (0.0, 1.0)

    lower_bound, upper_bound = mismatch_frequency_range

    for ref in references:
        ref_mask = adata.obs[ref_column] == ref
        ref_position_mask = adata.var.get(f"position_in_{ref}")
        if ref_position_mask is None:
            ref_position_mask = pd.Series(np.ones(n_vars, dtype=bool), index=adata.var.index)
        else:
            ref_position_mask = ref_position_mask.astype(bool)

        frequency_values = np.full(n_vars, np.nan, dtype=float)
        variable_flags = np.zeros(n_vars, dtype=bool)
        mismatch_base_frequencies: list[list[tuple[str, float]]] = [[] for _ in range(n_vars)]

        if ref_mask.sum() == 0:
            adata.var[f"{ref}_mismatch_frequency"] = pd.Series(
                frequency_values, index=adata.var.index
            )
            adata.var[f"{ref}_variable_sequence_site"] = pd.Series(
                variable_flags, index=adata.var.index
            )
            adata.var[f"{ref}_mismatch_base_frequencies"] = pd.Series(
                mismatch_base_frequencies, index=adata.var.index
            )
            continue

        mismatch_matrix = np.asarray(adata.layers[mismatch_layer][ref_mask])
        if has_span_mask:
            span_matrix = np.asarray(adata.layers[read_span_layer][ref_mask])
            coverage_mask = span_matrix > 0
            coverage_counts = coverage_mask.sum(axis=0).astype(float)
        else:
            coverage_mask = np.ones_like(mismatch_matrix, dtype=bool)
            coverage_counts = np.full(n_vars, ref_mask.sum(), dtype=float)

        mismatch_mask = (~np.isin(mismatch_matrix, [n_value, pad_value])) & coverage_mask
        mismatch_counts = mismatch_mask.sum(axis=0)

        frequency_values = np.divide(
            mismatch_counts,
            coverage_counts,
            out=np.full(n_vars, np.nan, dtype=float),
            where=coverage_counts > 0,
        )
        frequency_values = np.where(ref_position_mask.values, frequency_values, np.nan)

        variable_flags = (
            (frequency_values >= lower_bound)
            & (frequency_values <= upper_bound)
            & ref_position_mask.values
        )

        base_counts_by_int: dict[int, np.ndarray] = {}
        for base_int in base_int_to_label:
            base_counts_by_int[base_int] = ((mismatch_matrix == base_int) & coverage_mask).sum(
                axis=0
            )

        for idx in range(n_vars):
            if not ref_position_mask.iloc[idx] or coverage_counts[idx] == 0:
                continue
            base_freqs: list[tuple[str, float]] = []
            for base_int, base_label in base_int_to_label.items():
                count = base_counts_by_int[base_int][idx]
                if count > 0:
                    base_freqs.append((base_label, float(count / coverage_counts[idx])))
            mismatch_base_frequencies[idx] = base_freqs

        adata.var[f"{ref}_mismatch_frequency"] = pd.Series(frequency_values, index=adata.var.index)
        adata.var[f"{ref}_variable_sequence_site"] = pd.Series(
            variable_flags, index=adata.var.index
        )
        adata.var[f"{ref}_mismatch_base_frequencies"] = pd.Series(
            mismatch_base_frequencies, index=adata.var.index
        )

    adata.uns[uns_flag] = True

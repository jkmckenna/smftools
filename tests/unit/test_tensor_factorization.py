from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.tools.tensor_factorization import (
    build_sequence_one_hot_and_mask,
    calculate_sequence_cp_decomposition,
)


@pytest.mark.unit
def test_build_sequence_one_hot_and_mask() -> None:
    mapping = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
    encoded = np.array(
        [
            [mapping["A"], mapping["C"], mapping["N"], mapping["T"]],
            [mapping["G"], mapping["PAD"], mapping["T"], mapping["A"]],
        ],
        dtype=np.int16,
    )

    one_hot, mask = build_sequence_one_hot_and_mask(encoded)

    assert one_hot.shape == (2, 4, 4)
    assert mask.shape == (2, 4)
    assert mask.tolist() == [[True, True, False, True], [True, False, True, True]]
    assert one_hot[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert one_hot[0, 1].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert one_hot[0, 2].tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_calculate_sequence_cp_decomposition() -> None:
    pytest.importorskip("tensorly")
    mapping = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
    encoded = np.array(
        [
            [mapping["A"], mapping["C"], mapping["N"], mapping["T"]],
            [mapping["G"], mapping["PAD"], mapping["T"], mapping["A"]],
            [mapping["T"], mapping["G"], mapping["C"], mapping["A"]],
        ],
        dtype=np.int16,
    )

    adata = ad.AnnData(np.zeros((encoded.shape[0], encoded.shape[1])))
    adata.layers["sequence_integer_encoding"] = encoded

    result = calculate_sequence_cp_decomposition(
        adata,
        layer="sequence_integer_encoding",
        rank=2,
        n_iter_max=10,
        random_state=0,
    )

    assert "X_cp_sequence" in result.obsm
    assert result.obsm["X_cp_sequence"].shape == (3, 2)
    assert "H_cp_sequence" in result.varm
    assert result.varm["H_cp_sequence"].shape == (4, 2)
    assert "cp_sequence" in result.uns
    assert result.uns["cp_sequence"]["base_factors"].shape == (4, 2)
    assert result.uns["cp_sequence"]["backend"] == "numpy"

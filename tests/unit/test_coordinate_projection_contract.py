"""Storage is coordinate-frame invariant; reindexing is a display projection.

The contract: data is always stored in the reference coordinate frame, and
`reindexing_offsets` / `reindexing_invert` define a coordinate system for
position-dependent plots only. Two runs differing solely in display settings
must remain directly comparable, which is exactly what the superseded
`invert_adata` broke -- it flipped `X` and every layer, so an inverted store and
an uninverted store were different artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from smftools.preprocessing.reindex_references_adata import reindex_coordinates

pytestmark = pytest.mark.unit


def _cfg(tmp_path, body: str):
    from smftools.config.experiment_config import ExperimentConfig

    path = tmp_path / "experiment_config.csv"
    path.write_text("variable,value,help,options,type\n" + body, encoding="utf-8")
    return ExperimentConfig.from_csv(path)


def test_offset_is_applied_before_the_sign():
    """Anchor preservation. Sign-first would mirror about the origin instead."""
    positions = np.array([944, 2000, 3052, 3795])
    offsets = {"ref": -3052}
    plain = reindex_coordinates(positions, "ref", offsets, None)
    inverted = reindex_coordinates(positions, "ref", offsets, {"ref": True})
    assert list(plain) == [-2108, -1052, 0, 743]
    assert list(inverted) == [2108, 1052, 0, -743]
    # The anchor is fixed by the offset and survives inversion.
    assert plain[2] == inverted[2] == 0


def test_inversion_reverses_render_order_without_touching_data():
    """Ascending display order is descending genomic order when inverted.

    This is how the axis reverses without any array being reordered -- the
    property the whole projection model rests on.
    """
    positions = np.array([944, 2000, 3052, 3795])
    values = reindex_coordinates(positions, "ref", {"ref": -3052}, {"ref": True})
    order = np.argsort(values, kind="stable")
    assert list(positions[order]) == [3795, 3052, 2000, 944]


def test_invert_flag_accepts_bool_and_per_reference_mapping():
    positions = np.array([10, 20])
    assert list(reindex_coordinates(positions, "ref", None, True)) == [-10, -20]
    assert list(reindex_coordinates(positions, "ref", None, {"other": True})) == [10, 20]


def test_no_offset_and_no_invert_is_identity():
    positions = np.array([10, 20, 30])
    assert list(reindex_coordinates(positions, "ref", None, None)) == [10, 20, 30]


def test_invert_adata_is_rejected_with_a_migration_message(tmp_path):
    """Silence is the failure mode being fixed.

    Eight EMseq runs carried `invert_adata: True` and produced uninverted plots
    for months, while migrated DAFseq runs inverted correctly.
    """
    with pytest.raises(ValueError, match="reindexing_invert"):
        _cfg(tmp_path, "invert_adata,TRUE,,,bool\n")


def test_invert_adata_false_stays_silent(tmp_path):
    """Only the value that carries intent is rejected; the default is harmless."""
    cfg, _ = _cfg(tmp_path, "invert_adata,FALSE,,,bool\n")
    assert cfg.invert_adata is False


def test_reindexing_invert_is_accepted(tmp_path):
    cfg, _ = _cfg(tmp_path, "reindexing_invert,TRUE,,,bool\n")
    assert cfg.reindexing_invert is True

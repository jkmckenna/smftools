"""Demux status must survive into the *published* obs (`F31`).

`EGL-29` recovered single- vs double-ended barcode status from a FASTQ tree and
was covered by 45 tests across three lanes -- every one of which stopped at the
sidecar. None asked whether the value reached the obs that analysis reads, and
it did not: `demux_type` was only ever written on a dense-AnnData branch the
partitioned pipeline never takes, and `barcode_agreement` was computed on the
read frame and then dropped by `_RAW_SCALAR_OBS_COLUMNS`, which is an allowlist.

So these tests deliberately assert against a **published raw store**, not
against the functions that populate it. That is the check that was missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.demux_agreement import derive_demux_type_from_bm
from smftools.informatics.raw_store import _RAW_SCALAR_OBS_COLUMNS, write_raw_store

pytestmark = pytest.mark.unit


def _frame(bm_values):
    rows = []
    for index, bm in enumerate(bm_values):
        rows.append(
            {
                "read_id": f"read{index}",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "barcode": "bc1",
                "BC": "bc1",
                "BM": bm,
                "sample": "bc1",
                "reference_start": 0,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30, 30, 30, 30],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [1.0, np.nan, 0.0, 1.0],
                "read_length": 4,
                "mapped_length": 4,
                "reference_length": 12,
                "read_quality": 30,
                "mapping_quality": 60,
            }
        )
    return pd.DataFrame(rows)


# --- the derivation -----------------------------------------------------------


def test_bm_maps_to_demux_type():
    frame = _frame(["both", "read_start_only", "read_end_only", "mismatch", "unclassified"])
    derive_demux_type_from_bm(frame)
    assert list(frame["demux_type"]) == [
        "double",
        "single",
        "single",
        "unclassified",
        "unclassified",
    ]


def test_mapping_matches_the_dense_helper():
    """Two paths deriving the same field must not drift."""
    import inspect

    from smftools.informatics import h5ad_functions

    dense = inspect.getsource(h5ad_functions.add_demux_type_from_bm_tag)
    for value in ("left_only", "right_only", "read_start_only", "read_end_only"):
        assert value in dense
    frame = _frame(["left_only", "right_only"])
    derive_demux_type_from_bm(frame)
    assert set(frame["demux_type"]) == {"single"}


def test_provenance_and_confidence_are_recorded():
    frame = _frame(["both", "mismatch"])
    derive_demux_type_from_bm(frame)
    assert set(frame["demux_type_source"]) == {"bm_tag"}
    assert list(frame["demux_type_confidence"]) == [1.0, 0.0]


def test_a_frame_without_bm_is_left_alone():
    frame = _frame(["both"]).drop(columns=["BM"])
    assert derive_demux_type_from_bm(frame) == 0
    assert "demux_type" not in frame.columns


def test_case_and_whitespace_are_tolerated():
    frame = _frame([" Both ", "READ_START_ONLY"])
    derive_demux_type_from_bm(frame)
    assert list(frame["demux_type"]) == ["double", "single"]


# --- the allowlist, which is what actually dropped it -------------------------


@pytest.mark.parametrize(
    "column",
    ["demux_type", "demux_type_source", "demux_type_confidence", "barcode_agreement", "BM"],
)
def test_column_is_named_in_the_obs_allowlist(column):
    """`_RAW_SCALAR_OBS_COLUMNS` is an allowlist: unnamed columns are dropped."""
    assert column in _RAW_SCALAR_OBS_COLUMNS


# --- the check that was missing: read it back from a published store ----------


def test_demux_status_survives_into_a_published_raw_store(tmp_path):
    frame = _frame(["both", "both", "read_start_only", "mismatch"])
    derive_demux_type_from_bm(frame)

    outputs = write_raw_store(frame, tmp_path / "raw_outputs", reference_lengths={"ref_top": 12})
    obs = pd.read_parquet(outputs["obs"]) if "obs" in outputs else None
    if obs is None:
        import anndata as ad

        obs = ad.read_h5ad(outputs["spine"]).obs

    assert "demux_type" in obs.columns, "demux_type must reach the published obs"
    assert list(obs["demux_type"]) == ["double", "double", "single", "unclassified"]
    assert set(obs["demux_type_source"]) == {"bm_tag"}


def test_barcode_agreement_survives_into_a_published_raw_store(tmp_path):
    frame = _frame(["both", "both"])
    frame.loc[1, "BC"] = "bc2"  # disagrees with the assigned barcode
    from smftools.informatics.demux_agreement import report_barcode_agreement

    report_barcode_agreement(frame)

    outputs = write_raw_store(frame, tmp_path / "raw_outputs", reference_lengths={"ref_top": 12})
    import anndata as ad

    obs = ad.read_h5ad(outputs["spine"]).obs

    assert "barcode_agreement" in obs.columns
    assert set(obs["barcode_agreement"]) == {"agree", "disagree"}

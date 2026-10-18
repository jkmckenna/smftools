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
                "barcode_assigned": "bc1",
                "barcode_rederived": "bc1",
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
        # `F50`: both ends read and disagreed -- a positive observation, kept
        # apart from "no usable barcode seen".
        "mismatch",
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
    """Confidence is in the *label*, not in the barcode it implies (`F50`).

    `mismatch` scored 0.0 only because it used to be folded into
    `unclassified`. As its own type it is a confident observation -- both ends
    were read and they disagreed -- so only `unclassified`, where the tag told
    us nothing, remains low-confidence. Neither yields a usable barcode, and
    consumers select on `demux_type` rather than this field.
    """
    frame = _frame(["both", "mismatch", "unclassified"])
    derive_demux_type_from_bm(frame)
    assert set(frame["demux_type_source"]) == {"bm_tag"}
    assert list(frame["demux_type_confidence"]) == [1.0, 1.0, 0.0]


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
    assert list(obs["demux_type"]) == ["double", "double", "single", "mismatch"]
    assert set(obs["demux_type_source"]) == {"bm_tag"}


def test_barcode_agreement_survives_into_a_published_raw_store(tmp_path):
    frame = _frame(["both", "both"])
    frame.loc[1, "barcode_rederived"] = "bc2"  # disagrees with the assigned barcode
    from smftools.informatics.demux_agreement import report_barcode_agreement

    report_barcode_agreement(frame)

    outputs = write_raw_store(frame, tmp_path / "raw_outputs", reference_lengths={"ref_top": 12})
    import anndata as ad

    obs = ad.read_h5ad(outputs["spine"]).obs

    assert "barcode_agreement" in obs.columns
    assert set(obs["barcode_agreement"]) == {"agree", "disagree"}


# --- the sequencing summary on the partitioned raw path (`EGL-29c`) -------------

#: columns `_attach_obs_metadata` joins from BAM metrics
_METRICS = [
    "BM",
    "read_length",
    "mapped_length",
    "reference_length",
    "read_quality",
    "mapping_quality",
]


def _summary(path, rows):
    pd.DataFrame(
        rows,
        columns=["read_id", "barcode_arrangement", "barcode_front_score", "barcode_rear_score"],
    ).to_csv(path, sep="\t", index=False)
    return path


def test_summary_demux_reaches_a_published_raw_store_from_the_partitioned_path(tmp_path):
    """A run demultiplexed upstream has no end evidence in its BAM, so every
    read is ``unclassified`` unless the basecaller's summary fills it. The
    dense load did; the partitioned raw path -- every current run -- did not,
    so a fresh raw extraction of such a run lost its demux status entirely."""
    import pickle
    from types import SimpleNamespace

    from smftools.cli.raw_adata import (
        SUMMARY_DEMUX_ATTR,
        _attach_obs_metadata,
        prepare_summary_demux_status,
    )

    summary = _summary(
        tmp_path / "sequencing_summary_x.txt",
        [
            ("read0", "barcode01", 99.0, 95.0),  # double
            ("read1", "barcode01", 99.0, 40.0),  # single
            ("read2", "barcode01", 30.0, 20.0),  # unclassified
        ],
    )
    cfg = SimpleNamespace(
        sequencing_summary_path=str(summary),
        barcode_end_score_threshold=62.0,
        experiment_name="exp",
        samtools_backend="python",
    )
    cache = tmp_path / "raw_outputs" / "intermediates" / "sequencing-summary-demux"
    table = prepare_summary_demux_status(cfg, None, cache)
    assert table is not None and table.parent == cache
    assert prepare_summary_demux_status(cfg, None, cache) == table  # read once, reused
    cfg = pickle.loads(pickle.dumps(cfg))  # what a worker process receives
    assert getattr(cfg, SUMMARY_DEMUX_ATTR) == str(table)

    frame = _frame(["unclassified"] * 3).drop(columns=_METRICS)
    metrics = {f"read{i}": (4, 30.0, 12, 4, 60, 0, 4) for i in range(3)}
    frame = _attach_obs_metadata(
        frame,
        cfg=cfg,
        bam_path=tmp_path / "unused.bam",
        barcode_sidecar=None,
        umi_sidecar=None,
        metrics=metrics,
    )
    outputs = write_raw_store(
        frame.reset_index(drop=True), tmp_path / "raw_outputs", reference_lengths={"ref_top": 12}
    )
    import anndata as ad

    obs = ad.read_h5ad(outputs["spine"]).obs
    assert list(obs.sort_index()["demux_type"].astype(str)) == ["double", "single", "unclassified"]


def test_an_existing_end_call_is_not_overridden_by_the_summary(tmp_path):
    from types import SimpleNamespace

    from smftools.cli.raw_adata import _attach_obs_metadata, prepare_summary_demux_status

    summary = _summary(tmp_path / "sequencing_summary_x.txt", [("read0", "barcode01", 99.0, 40.0)])
    cfg = SimpleNamespace(
        sequencing_summary_path=str(summary), experiment_name="exp", samtools_backend="python"
    )
    prepare_summary_demux_status(cfg, None, tmp_path / "cache")
    frame = _frame(["both"]).drop(columns=_METRICS).assign(demux_type="double")
    frame = _attach_obs_metadata(
        frame,
        cfg=cfg,
        bam_path=tmp_path / "unused.bam",
        barcode_sidecar=None,
        umi_sidecar=None,
        metrics={"read0": (4, 30.0, 12, 4, 60, 0, 4)},
    )
    assert list(frame["demux_type"]) == ["double"]  # the summary would say single


def test_no_summary_leaves_demux_as_derived(tmp_path):
    from types import SimpleNamespace

    from smftools.cli.raw_adata import SUMMARY_DEMUX_ATTR, prepare_summary_demux_status

    cfg = SimpleNamespace(sequencing_summary_path=None)
    assert prepare_summary_demux_status(cfg, tmp_path, tmp_path / "cache") is None
    assert not hasattr(cfg, SUMMARY_DEMUX_ATTR)

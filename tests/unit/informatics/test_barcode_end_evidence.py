"""Per-end barcode evidence reaches the published obs (`BQC-01`).

Barcode QC has to tell an end that carried no barcode from a barcode that was
not called, and a miscall from a real tag. That needs, per read end and in read
(sequencing) orientation: the sequence beyond the alignment, the match's edit
distance and where the barcode meets the insert. These tests follow each value
from where it is computed to the published raw store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.bam_functions import (
    FlankingConfig,
    PerEndFlankingConfig,
    _extract_and_match_barcodes_for_reads,
    _reverse_complement,
)
from smftools.informatics.ragged_store import (
    FRONT_CLIP_LENGTH,
    HAS_SUPPLEMENTARY_ALIGNMENT,
    REAR_CLIP_LENGTH,
    alignment_to_ragged_record,
    cigar_end_clips,
)
from smftools.informatics.raw_store import _RAW_SCALAR_OBS_COLUMNS, write_raw_store

pytestmark = pytest.mark.unit

EVIDENCE_COLUMNS = (
    "barcode_front_edit_distance",
    "barcode_rear_edit_distance",
    "barcode_front_insert_offset",
    "barcode_rear_insert_offset",
    FRONT_CLIP_LENGTH,
    REAR_CLIP_LENGTH,
    HAS_SUPPLEMENTARY_ALIGNMENT,
)


# --- clips in read orientation ------------------------------------------------


@pytest.mark.parametrize(
    "cigar, reverse, expected",
    [
        ("70S100M5S", False, (70, 5)),
        # A reverse record stores the read reverse-complemented: its leading
        # clip is the read's rear end.
        ("70S100M5S", True, (5, 70)),
        ("10H60S100M", False, (70, 0)),
        # Internal insertions are inside the aligned span, never an end clip.
        ("100M30I100M", False, (0, 0)),
        ("*", False, (0, 0)),
    ],
)
def test_end_clips_are_in_read_orientation(cigar, reverse, expected):
    assert cigar_end_clips(cigar, reverse) == expected


class _Read:
    is_unmapped = False
    is_secondary = False
    is_supplementary = False
    is_reverse = True
    query_name = "read1"
    reference_name = "chr1"
    reference_start = 1
    cigarstring = "2S4M3S"
    query_sequence = "NNAGCTNNN"
    query_qualities = [30] * 9

    def __init__(self, tags=()):
        self._tags = set(tags)

    def has_tag(self, tag):
        return tag in self._tags


def test_record_carries_end_clips_and_supplementary_flag():
    record = alignment_to_ragged_record(_Read(tags={"SA"}), "AAGCTGGT")
    assert (record[FRONT_CLIP_LENGTH], record[REAR_CLIP_LENGTH]) == (3, 2)
    assert record[HAS_SUPPLEMENTARY_ALIGNMENT] is True
    assert alignment_to_ragged_record(_Read(), "AAGCTGGT")[HAS_SUPPLEMENTARY_ALIGNMENT] is False


# --- barcode / insert boundary --------------------------------------------------


ADAPTER = "TTGACCAGTAGGCA"
BARCODES = {"NB01": "AAGGTTAA", "NB02": "CCTTGGCC"}


def _extract(sequence):
    flanking = PerEndFlankingConfig(left_ref_end=FlankingConfig(adapter_side=ADAPTER))
    (row,) = _extract_and_match_barcodes_for_reads(
        [sequence],
        barcode_length=8,
        barcode_search_window=60,
        barcode_max_edit_distance=1,
        barcode_adapter_matcher="exact",
        barcode_composite_max_edits=0,
        barcode_min_separation=None,
        require_both_ends=False,
        min_barcode_score=None,
        bc_refs=BARCODES,
        flanking_config=flanking,
        check_start=True,
        check_end=True,
    )
    return row


def test_boundary_offsets_count_bases_from_each_read_end():
    insert = "GATTACAGATTACAGATTACAGATTACA"
    tail = "GG"
    read = (
        ADAPTER
        + BARCODES["NB01"]
        + insert
        + _reverse_complement(BARCODES["NB02"])
        + _reverse_complement(ADAPTER)
        + tail
    )
    row = _extract(read)
    assert (row["B5"], row["B6"], row["BM"]) == ("NB01", "NB02", "mismatch")
    assert row["B7"] == len(ADAPTER) + 8
    assert row["B8"] == 8 + len(ADAPTER) + len(tail)
    # Insert recovered exactly from the two offsets.
    assert read[row["B7"] : len(read) - row["B8"]] == insert


def test_no_rear_flank_leaves_the_rear_boundary_empty():
    row = _extract(ADAPTER + BARCODES["NB01"] + "GATTACAGATTACAGATTACA")
    assert row["B7"] == len(ADAPTER) + 8
    assert row["B8"] is None and row["B6"] is None


# --- identity sidecar ---------------------------------------------------------


def test_identity_sidecar_names_the_per_end_evidence(tmp_path):
    pysam = pytest.importorskip("pysam")
    from smftools.informatics.barcode_sidecar import publish_barcode_identity_sidecar

    bam = tmp_path / "reads.bam"
    with pysam.AlignmentFile(str(bam), "wb", header={"HD": {"VN": "1.6"}, "SQ": []}) as out:
        for name in ("new", "legacy"):
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = "ACGT"
            read.query_qualities = pysam.qualitystring_to_array("IIII")
            read.is_unmapped = True
            out.write(read)
    classifier = tmp_path / "classifier.parquet"
    pd.DataFrame(
        {
            "read_name": ["new", "legacy"],
            "BC": ["NB01", "NB01"],
            "BM": ["both", "both"],
            "B1": [0.0, 1.0],
            "B2": [3.0, 2.0],
            "B5": ["NB01", "NB01"],
            "B6": ["NB01", "NB01"],
            # A sidecar written before `BQC-01` has no boundaries.
            "B7": [76.0, None],
            "B8": [51.0, None],
        }
    ).to_parquet(classifier, index=False)

    sidecar, _report = publish_barcode_identity_sidecar(
        bam, tmp_path / "identity.parquet", classifier_sidecar=classifier
    )
    rows = pd.read_parquet(sidecar).set_index("read_name")
    assert rows.loc["new", "barcode_front_edit_distance"] == 0
    assert rows.loc["new", "barcode_rear_edit_distance"] == 3
    assert rows.loc["new", "barcode_front_insert_offset"] == 76
    assert rows.loc["new", "barcode_rear_insert_offset"] == 51
    assert np.isnan(rows.loc["legacy", "barcode_rear_insert_offset"])


# --- the published store ------------------------------------------------------


@pytest.mark.parametrize("column", EVIDENCE_COLUMNS)
def test_column_is_named_in_the_obs_allowlist(column):
    """`_RAW_SCALAR_OBS_COLUMNS` is an allowlist: unnamed columns are dropped."""
    assert column in _RAW_SCALAR_OBS_COLUMNS


def test_end_evidence_survives_into_a_published_raw_store(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "read_id": f"read{index}",
                "reference": "ref",
                "Reference_strand": "ref_top",
                "barcode": "bc1",
                "sample": "bc1",
                "reference_start": 0,
                "cigar": "4M",
                "aligned_length": 4,
                "sequence": [0, 1, 2, 3],
                "quality": [30, 30, 30, 30],
                "mismatch": [4, 4, 4, 4],
                "modification_signal": [1.0, np.nan, 0.0, 1.0],
                "barcode_front_edit_distance": 0.0,
                "barcode_rear_edit_distance": np.nan,
                "barcode_front_insert_offset": 76.0,
                "barcode_rear_insert_offset": np.nan,
                FRONT_CLIP_LENGTH: 76,
                REAR_CLIP_LENGTH: rear,
                HAS_SUPPLEMENTARY_ALIGNMENT: False,
            }
            for index, rear in enumerate((51, 0))
        ]
    )
    outputs = write_raw_store(frame, tmp_path / "raw_outputs", reference_lengths={"ref_top": 12})
    obs = pd.read_parquet(outputs["obs"]) if "obs" in outputs else None
    if obs is None:
        import anndata as ad

        obs = ad.read_h5ad(outputs["spine"]).obs
    for column in EVIDENCE_COLUMNS:
        assert column in obs.columns, f"{column} must reach the published obs"
    assert list(obs[REAR_CLIP_LENGTH]) == [51, 0]
    assert list(obs["barcode_front_insert_offset"]) == [76, 76]

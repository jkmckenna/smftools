import numpy as np
import pandas as pd

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.informatics import bam_functions
from smftools.informatics.ragged_store import materialize_ragged


def test_extract_base_identities_returns_mismatch_integer_encoding(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self):
            self.is_mapped = True
            self.is_reverse = False
            self.query_name = "read1"
            self.query_sequence = "AGGT"
            self.query_qualities = [30, 31, 32, 33]
            self.reference_start = 0
            self.reference_end = 4

        def get_aligned_pairs(self, matches_only=True):
            return [(0, 0), (1, 1), (2, 2), (3, 3)]

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            self.mapped = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, chromosome):
            return iter([FakeRead()])

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")
    sequence = "ACGT"

    (
        _fwd,
        _rev,
        _mismatch_counts,
        _mismatch_trends,
        mismatch_base_identities,
        base_quality_scores,
        read_span_masks,
    ) = bam_functions.extract_base_identities(
        bam_path,
        "chr1",
        range(len(sequence)),
        max_reference_length=6,
        sequence=sequence,
        samtools_backend="python",
    )

    mismatch_array = mismatch_base_identities["read1"]
    expected = np.array(
        [
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["G"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"],
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"],
        ],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(mismatch_array, expected)

    quality_array = base_quality_scores["read1"]
    expected_quality = np.array([30, 31, 32, 33, -1, -1], dtype=np.int16)
    np.testing.assert_array_equal(quality_array, expected_quality)

    span_mask = read_span_masks["read1"]
    expected_span = np.array([1, 1, 1, 1, 0, 0], dtype=np.int8)
    np.testing.assert_array_equal(span_mask, expected_span)


def test_read_relative_extraction_matches_legacy_dense_layers(monkeypatch, tmp_path):
    class FakeRead:
        is_mapped = True
        is_unmapped = False
        is_secondary = False
        is_supplementary = False
        is_reverse = False
        query_name = "read1"
        query_sequence = "AGGT"
        query_qualities = [30, 31, 32, 33]
        reference_name = "chr1"
        reference_start = 0
        reference_end = 4
        cigarstring = "4M"

        def get_aligned_pairs(self, matches_only=True):
            return [(0, 0), (1, 1), (2, 2), (3, 3)]

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            self.mapped = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, chromosome):
            return iter([FakeRead()])

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )
    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")
    reference = "ACGT"

    legacy = bam_functions.extract_base_identities(
        bam_path,
        "chr1",
        range(4),
        max_reference_length=4,
        sequence=reference,
        samtools_backend="python",
    )
    records = bam_functions.extract_read_relative_base_identities(
        bam_path,
        "chr1",
        reference,
        samtools_backend="python",
    )
    dense = materialize_ragged(
        pd.DataFrame(records),
        obs=pd.DataFrame({"Reference_strand": ["chr1_top"]}, index=["read1"]),
        reference_lengths={"chr1_top": 4},
    )

    legacy_sequence = np.array(
        [MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[base.decode()] for base in legacy[0]["read1"]],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(dense.layers["sequence_integer_encoding"][0], legacy_sequence)
    np.testing.assert_array_equal(dense.layers["mismatch_integer_encoding"][0], legacy[4]["read1"])
    np.testing.assert_array_equal(dense.layers["base_quality_scores"][0], legacy[5]["read1"])
    np.testing.assert_array_equal(dense.layers["read_span_mask"][0], legacy[6]["read1"])

import numpy as np

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.informatics import bam_functions


def test_extract_base_identities_returns_mismatch_integer_encoding(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self):
            self.is_mapped = True
            self.is_reverse = False
            self.query_name = "read1"
            self.query_sequence = "AGGT"

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

from smftools.informatics import bam_functions


def test_find_secondary_supplementary_read_names_python(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self, name, secondary=False, supplementary=False):
            self.query_name = name
            self.is_secondary = secondary
            self.is_supplementary = supplementary

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, until_eof=True):
            return iter(
                [
                    FakeRead("read1", secondary=True),
                    FakeRead("read2", supplementary=True),
                    FakeRead("read3", secondary=True, supplementary=True),
                ]
            )

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    secondary, supplementary = bam_functions.find_secondary_supplementary_read_names(
        bam_path,
        ["read1", "read2", "read3", "read4"],
        samtools_backend="python",
    )

    assert secondary == {"read1", "read3"}
    assert supplementary == {"read2", "read3"}


def test_extract_secondary_supplementary_alignment_spans_python(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(
            self,
            name,
            secondary=False,
            supplementary=False,
            reference_start=10,
            reference_end=30,
            read_span=20,
        ):
            self.query_name = name
            self.is_secondary = secondary
            self.is_supplementary = supplementary
            self.reference_start = reference_start
            self.reference_end = reference_end
            self.query_alignment_length = read_span

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, until_eof=True):
            return iter(
                [
                    FakeRead("read1", secondary=True, reference_start=5, reference_end=25, read_span=20),
                    FakeRead(
                        "read2", supplementary=True, reference_start=100, reference_end=140, read_span=40
                    ),
                    FakeRead(
                        "read3",
                        secondary=True,
                        supplementary=True,
                        reference_start=200,
                        reference_end=220,
                        read_span=20,
                    ),
                ]
            )

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    secondary_spans, supplementary_spans = (
        bam_functions.extract_secondary_supplementary_alignment_spans(
            bam_path,
            ["read1", "read2", "read3", "read4"],
            samtools_backend="python",
        )
    )

    assert secondary_spans == {
        "read1": [(5.0, 25.0, 20.0)],
        "read3": [(200.0, 220.0, 20.0)],
    }
    assert supplementary_spans == {
        "read2": [(100.0, 140.0, 40.0)],
        "read3": [(200.0, 220.0, 20.0)],
    }

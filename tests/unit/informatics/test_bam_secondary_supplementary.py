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

from smftools.informatics import bam_functions


def test_extract_read_tags_from_bam_python(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self):
            self.query_name = "read1"
            self.flag = 99
            self.cigarstring = "4M"
            self._tags = {"NM": 1, "MD": "4", "MM": "C+m,0;", "ML": [200]}

        def get_tag(self, tag):
            return self._tags[tag]

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, until_eof=True):
            return iter([FakeRead()])

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    read_tags = bam_functions.extract_read_tags_from_bam(
        bam_path,
        tag_names=["NM", "MD", "MM", "ML"],
        include_flags=True,
        include_cigar=True,
        samtools_backend="python",
    )

    assert read_tags["read1"]["CIGAR"] == "4M"
    assert "proper_pair" in read_tags["read1"]["FLAGS"]
    assert read_tags["read1"]["NM"] == 1
    assert read_tags["read1"]["MD"] == "4"
    assert read_tags["read1"]["MM"] == "C+m,0;"
    assert read_tags["read1"]["ML"] == [200]
